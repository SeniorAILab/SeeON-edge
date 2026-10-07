from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from threading import TIMEOUT_MAX, Condition, get_ident
from time import monotonic
from typing import TypeVar

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout, TooManyRequests

from shared.boundary import register_translation_target

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class PoolBudget:
    max_connections: int
    max_waiting: int
    acquire_timeout_sec: float
    statement_timeout_ms: int
    lock_timeout_ms: int
    startup_timeout_sec: float

    def __post_init__(self) -> None:
        for name in (
            "max_connections",
            "max_waiting",
            "statement_timeout_ms",
            "lock_timeout_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 2_147_483_647:
                raise ValueError(f"{name} must be a positive 32-bit integer")
        for name in ("acquire_timeout_sec", "startup_timeout_sec"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= 2_147_483_647:
                raise ValueError(f"{name} must be positive and finite")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")


class PostgresError(RuntimeError):
    ...


register_translation_target(PostgresError)


class PostgresStartupError(PostgresError):
    ...


class PostgresShutdownError(PostgresError):
    ...


class PostgresShutdownTimeout(PostgresShutdownError):
    ...


class PostgresUnavailable(PostgresError):
    ...


class PostgresPoolBusy(PostgresUnavailable):
    ...


class PostgresTransactionStateError(PostgresError):
    ...


class CommitOutcomeUnknown(PostgresError):
    def __init__(self) -> None:
        super().__init__("PostgreSQL commit outcome is unknown; automatic retry is forbidden")


class _ShutdownFailure(Enum):
    FINALIZER = "PostgreSQL shutdown finalizer failed"
    CANCELLED_FINALIZER = "PostgreSQL shutdown finalizer was cancelled"
    UNKNOWN_COMMIT = "PostgreSQL shutdown finalizer commit outcome is unknown"
    FINALIZER_DEADLINE = "PostgreSQL shutdown finalizer exceeded its deadline"
    POOL = "PostgreSQL shutdown pool cleanup failed"
    CANCELLED_POOL = "PostgreSQL shutdown pool cleanup was cancelled"
    POOL_DEADLINE = "PostgreSQL shutdown pool cleanup exceeded its deadline"


class _PrivateConnection(psycopg.Connection):
    def __repr__(self) -> str:
        return "<PostgresConnection redacted>"

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs) -> _PrivateConnection:
        try:
            return super().connect(conninfo, **kwargs)
        except (psycopg.Error, OSError, ValueError, TypeError):
            raise psycopg.OperationalError("PostgreSQL connection failed") from None


class PostgresDatabase:
    def __init__(self, conninfo: str, schema: str, budget: PoolBudget) -> None:
        if not isinstance(conninfo, str):
            raise TypeError("PostgreSQL connection information must be text")
        if not conninfo.strip() or "\x00" in conninfo:
            raise ValueError("PostgreSQL connection information must be nonblank without NUL bytes")
        if (
            not isinstance(schema, str)
            or not schema
            or "\x00" in schema
            or len(schema.encode("utf-8")) > 63
        ):
            raise ValueError("PostgreSQL schema must be a nonempty identifier of at most 63 bytes")
        self._schema = schema
        self._budget = budget
        self._condition = Condition()
        self._started = False
        self._starting_thread: int | None = None
        self._stopped = False
        self._closing = False
        self._active = 0
        self._depths: dict[int, int] = {}
        self._pool_disposed = False
        self._shutdown_complete = False
        self._shutdown_failure: _ShutdownFailure | None = None
        self._finalizer_attempted = False
        self._finalizer_thread: int | None = None
        self._finalizer_write_available = False
        try:
            self._pool = ConnectionPool(
                conninfo=conninfo,
                connection_class=_PrivateConnection,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": max(2, math.ceil(budget.startup_timeout_sec)),
                    "application_name": "seeon-edge",
                },
                min_size=1,
                max_size=budget.max_connections,
                max_waiting=budget.max_waiting,
                timeout=budget.acquire_timeout_sec,
                num_workers=1,
                open=False,
                name="seeon-postgres",
                configure=self._configure,
                reconnect_timeout=budget.startup_timeout_sec,
            )
        except (psycopg.Error, OSError, ValueError, TypeError):
            raise PostgresStartupError("PostgreSQL pool configuration failed") from None

    def __repr__(self) -> str:
        return "<PostgresDatabase redacted>"

    @property
    def schema(self) -> str:
        return self._schema

    def _configure(self, connection: psycopg.Connection) -> None:
        try:
            connection.execute(
                "SELECT pg_catalog.set_config('TimeZone', 'UTC', false), "
                "pg_catalog.set_config('synchronous_commit', 'on', false), "
                "pg_catalog.set_config('statement_timeout', %s, false), "
                "pg_catalog.set_config('lock_timeout', %s, false), "
                "pg_catalog.set_config('idle_in_transaction_session_timeout', %s, false)",
                (
                    str(self._budget.statement_timeout_ms),
                    str(self._budget.lock_timeout_ms),
                    str(self._budget.statement_timeout_ms),
                ),
            )
            connection.execute(
                sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(
                    sql.Identifier(self._schema)
                )
            )
            row = connection.execute(
                "SELECT pg_catalog.current_setting('fsync'), "
                "pg_catalog.current_setting('full_page_writes'), "
                "pg_catalog.current_setting('synchronous_commit'), "
                "pg_catalog.current_schema(), "
                "pg_catalog.current_setting('server_encoding'), "
                "pg_catalog.current_setting('session_replication_role')"
            ).fetchone()
        except (psycopg.Error, OSError, ValueError, TypeError):
            raise PostgresStartupError("PostgreSQL connection configuration is unsafe") from None
        if row != ("on", "on", "on", self._schema, "UTF8", "origin"):
            raise PostgresStartupError("PostgreSQL durability or namespace configuration is unsafe")

    @staticmethod
    def _require_idle_session(connection: psycopg.Connection) -> None:
        if (
            not connection.autocommit
            or connection.info.transaction_status is not TransactionStatus.IDLE
        ):
            raise PostgresTransactionStateError("PostgreSQL checkout session is not idle")

    def _configure_checkout(self, connection: psycopg.Connection) -> None:
        try:
            self._require_idle_session(connection)
            self._configure(connection)
        except BaseException as error:
            try:
                connection.close()
            except BaseException as cleanup:
                failure = self._primary_failure(error, cleanup)
                if failure is cleanup:
                    raise cleanup from None
                raise failure from None
            raise

    def _validate_return(
        self, connection: psycopg.Connection, primary_error: BaseException | None
    ) -> None:
        if connection.closed:
            return
        try:
            self._configure_checkout(connection)
        except BaseException as error:
            failure = self._primary_failure(primary_error, error)
            if failure is error:
                raise error from None
            raise failure from None

    def start(self) -> None:
        with self._condition:
            if self._stopped:
                raise PostgresStartupError("PostgreSQL database owner is closed")
            if self._started:
                return
            if self._starting_thread is not None:
                raise PostgresStartupError("PostgreSQL startup is already in progress")
            self._starting_thread = get_ident()
        primary_error: BaseException | None = None
        try:
            self._pool.open(wait=False)
            with self._pool.connection(timeout=self._budget.startup_timeout_sec) as connection:
                try:
                    self._configure_checkout(connection)
                except BaseException as error:
                    primary_error = error
                    raise
            self._complete_startup()
        except BaseException as error:
            failure = self._primary_failure(primary_error, error)
            if failure is error and not isinstance(error, Exception):
                raise error from None
            if not isinstance(failure, Exception):
                raise failure from None
            raise PostgresStartupError("PostgreSQL bounded startup failed") from None
        finally:
            with self._condition:
                if not self._started:
                    self._stopped = True
                self._starting_thread = None
                cleanup = not self._started and not self._closing
                if cleanup:
                    self._closing = True
                self._condition.notify_all()
            if cleanup:
                try:
                    with suppress(BaseException):
                        self._close_pool(monotonic() + self._budget.startup_timeout_sec)
                finally:
                    with self._condition:
                        self._closing = False
                        self._condition.notify_all()

    def _complete_startup(self) -> None:
        with self._condition:
            if self._stopped:
                raise PostgresStartupError("PostgreSQL database owner is closed")
            self._started = True

    def stop_admission(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise PostgresShutdownTimeout("PostgreSQL shutdown deadline exhausted")
        return remaining

    def close(self, *, timeout_sec: float, finalizer: Callable[[], None] | None = None) -> None:
        try:
            valid_timeout = (
                type(timeout_sec) in (int, float) and timeout_sec > 0 and math.isfinite(timeout_sec)
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("timeout_sec must be finite and positive")
        deadline = monotonic() + timeout_sec
        thread = get_ident()
        with self._condition:
            if self._depths.get(thread, 0) or thread in (
                self._finalizer_thread,
                self._starting_thread,
            ):
                raise PostgresShutdownError("PostgreSQL shutdown cannot drain its own scope")
            if self._closing:
                raise PostgresShutdownError("PostgreSQL shutdown is already in progress")
            if self._shutdown_complete:
                return
            self._stopped = True
            self._closing = True
        try:
            with self._condition:
                while self._active or self._starting_thread is not None:
                    self._condition.wait(timeout=min(self._remaining(deadline), TIMEOUT_MAX))
            self._remaining(deadline)
            try:
                self._finalize(finalizer, deadline)
            except BaseException:
                with suppress(BaseException):
                    self._close_pool(deadline)
                raise
            try:
                self._close_pool(deadline)
            except Exception:
                self._raise_shutdown_failure()
                raise
            self._raise_shutdown_failure()
            self._remaining(deadline)
            with self._condition:
                self._shutdown_complete = True
        finally:
            with self._condition:
                self._closing = False
                self._condition.notify_all()

    def _record_finalizer_failure(self, error: BaseException) -> None:
        if isinstance(error, CommitOutcomeUnknown):
            failure = _ShutdownFailure.UNKNOWN_COMMIT
        elif not isinstance(error, Exception):
            failure = _ShutdownFailure.CANCELLED_FINALIZER
        elif isinstance(error, PostgresShutdownTimeout):
            failure = _ShutdownFailure.FINALIZER_DEADLINE
        else:
            failure = _ShutdownFailure.FINALIZER
        with self._condition:
            if self._shutdown_failure is None:
                self._shutdown_failure = failure

    def _raise_shutdown_failure(self) -> None:
        with self._condition:
            failure = self._shutdown_failure
        if failure is _ShutdownFailure.UNKNOWN_COMMIT:
            raise CommitOutcomeUnknown() from None
        if failure in (_ShutdownFailure.FINALIZER_DEADLINE, _ShutdownFailure.POOL_DEADLINE):
            raise PostgresShutdownTimeout(failure.value) from None
        if failure is not None:
            raise PostgresShutdownError(failure.value) from None

    def _finalize(self, finalizer: Callable[[], None] | None, deadline: float) -> None:
        with self._condition:
            if (
                finalizer is None
                or not self._started
                or self._pool_disposed
                or self._finalizer_attempted
                or self._shutdown_failure is not None
            ):
                return
            self._remaining(deadline)
            self._finalizer_attempted = True
            self._finalizer_thread = get_ident()
            self._finalizer_write_available = True
        try:
            finalizer()
            self._remaining(deadline)
            self._raise_shutdown_failure()
        except BaseException as error:
            self._record_finalizer_failure(error)
            if not isinstance(error, Exception):
                raise error from None
            self._raise_shutdown_failure()
        finally:
            with self._condition:
                self._finalizer_thread = None
                self._finalizer_write_available = False

    def _close_pool(self, deadline: float) -> None:
        with self._condition:
            assert self._closing and not self._active and self._starting_thread is None
            assert self._finalizer_thread is None
            if self._pool_disposed:
                return
        remaining = self._remaining(deadline)
        try:
            self._pool.close(timeout=min(remaining, TIMEOUT_MAX))
        except BaseException as error:
            with self._condition:
                if self._shutdown_failure is None:
                    self._shutdown_failure = (
                        _ShutdownFailure.POOL
                        if isinstance(error, Exception)
                        else _ShutdownFailure.CANCELLED_POOL
                    )
            if not isinstance(error, Exception):
                raise error from None
            raise PostgresShutdownError("PostgreSQL shutdown pool cleanup failed") from None
        with self._condition:
            self._pool_disposed = True
        try:
            self._remaining(deadline)
        except PostgresShutdownTimeout:
            with self._condition:
                if self._shutdown_failure is None:
                    self._shutdown_failure = _ShutdownFailure.POOL_DEADLINE
            raise

    def stats(self) -> dict[str, int]:
        return self._pool.get_stats()

    def _admit(self, *, write: bool) -> bool:
        thread = get_ident()
        with self._condition:
            depth = self._depths.get(thread, 0)
            privileged = (
                write
                and self._finalizer_thread == thread
                and self._finalizer_write_available
                and not depth
            )
            if not (self._started and not self._stopped) and not privileged:
                raise PostgresUnavailable("PostgreSQL database owner is not running")
            if privileged:
                self._finalizer_write_available = False
            self._active += 1
            self._depths[thread] = depth + 1
            return privileged

    def _release(self) -> None:
        thread = get_ident()
        with self._condition:
            depth = self._depths[thread] - 1
            if depth:
                self._depths[thread] = depth
            else:
                del self._depths[thread]
            self._active -= 1
            self._condition.notify_all()

    @staticmethod
    def _rollback(connection: psycopg.Connection) -> None:
        try:
            connection.rollback()
        except (psycopg.Error, OSError):
            connection.close()

    @staticmethod
    def _require_active_transaction(connection: psycopg.Connection) -> None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise PostgresTransactionStateError(
                "PostgreSQL callback did not leave an active, successful transaction"
            )

    @staticmethod
    def _primary_failure(primary: BaseException | None, cleanup: BaseException) -> BaseException:
        if primary is None:
            return cleanup
        if isinstance(primary, CommitOutcomeUnknown) or not isinstance(primary, Exception):
            return primary
        return cleanup if not isinstance(cleanup, Exception) else primary

    def _run(
        self, callback: Callable[[psycopg.Connection], _Result], *, begin: str, write: bool = False
    ) -> _Result:
        privileged = self._admit(write=write)
        primary_error: BaseException | None = None
        try:
            try:
                with self._pool.connection(timeout=self._budget.acquire_timeout_sec) as connection:
                    try:
                        self._configure_checkout(connection)
                        connection.execute(begin)
                        try:
                            result = callback(connection)
                            self._require_active_transaction(connection)
                        except BaseException as error:
                            primary_error = error
                            self._rollback(connection)
                            raise
                        try:
                            connection.commit()
                        except (psycopg.OperationalError, psycopg.InterfaceError):
                            primary_error = CommitOutcomeUnknown()
                            connection.close()
                            raise primary_error from None
                    except BaseException as error:
                        primary_error = self._primary_failure(primary_error, error)
                        raise
                    finally:
                        try:
                            self._validate_return(connection, primary_error)
                        except BaseException as error:
                            primary_error = self._primary_failure(primary_error, error)
                            raise
            except BaseException as error:
                failure = self._primary_failure(primary_error, error)
                if isinstance(failure, (PoolTimeout, TooManyRequests)):
                    raise PostgresPoolBusy("PostgreSQL acquisition budget exhausted") from None
                if isinstance(
                    failure, (PoolClosed, psycopg.OperationalError, psycopg.InterfaceError)
                ):
                    raise PostgresUnavailable("PostgreSQL connection unavailable") from None
                if failure is error:
                    raise error from None
                raise failure from None
            else:
                return result
        except BaseException as error:
            if privileged:
                self._record_finalizer_failure(error)
            raise
        finally:
            self._release()

    def read(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        return self._run(callback, begin="BEGIN ISOLATION LEVEL READ COMMITTED, READ ONLY")

    def read_snapshot(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        return self._run(callback, begin="BEGIN ISOLATION LEVEL REPEATABLE READ, READ ONLY")

    def transact(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        return self._run(
            callback, begin="BEGIN ISOLATION LEVEL READ COMMITTED, READ WRITE", write=True
        )


__all__ = [
    "CommitOutcomeUnknown",
    "PoolBudget",
    "PostgresDatabase",
    "PostgresError",
    "PostgresPoolBusy",
    "PostgresShutdownError",
    "PostgresShutdownTimeout",
    "PostgresStartupError",
    "PostgresTransactionStateError",
    "PostgresUnavailable",
]
