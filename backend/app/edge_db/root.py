from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, require_authority
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase, PostgresError

API_POSTGRES_DSN_FILE_ENV = "API_POSTGRES_DSN_FILE"
API_POSTGRES_AUTHORITY_FILE_ENV = "API_POSTGRES_AUTHORITY_FILE"
API_POSTGRES_SCHEMA_ENV = "API_POSTGRES_SCHEMA"
DEFAULT_POSTGRES_SCHEMA = "seeon_edge"
DEFAULT_POOL_BUDGET = PoolBudget(
    max_connections=8,
    max_waiting=32,
    acquire_timeout_sec=2.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=10.0,
)
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 10.0


class PostgresRootError(RuntimeError):
    ...


@dataclass(frozen=True, slots=True)
class PostgresRoot:
    database: PostgresDatabase
    authority: AuthorityToken


def _required_file(environ: Mapping[str, str], name: str, label: str) -> str:
    raw_path = environ.get(name, "").strip()
    if not raw_path:
        raise PostgresRootError(f"{label} file is not configured; set {name}")
    try:
        text = Path(raw_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        text = None
    if text is None:
        raise PostgresRootError(f"{label} file named by {name} is unreadable")
    if not text:
        raise PostgresRootError(f"{label} file named by {name} is empty")
    return text


def _parse_authority(text: str) -> AuthorityToken:
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    token: AuthorityToken | None = None
    if isinstance(payload, dict) and set(payload) == {"generation", "writer_token"}:
        try:
            token = AuthorityToken(
                generation=payload["generation"], writer_token=UUID(str(payload["writer_token"]))
            )
        except (ValueError, TypeError):
            token = None
    if token is None:
        raise PostgresRootError(
            f"persistence authority file named by {API_POSTGRES_AUTHORITY_FILE_ENV} is invalid"
        )
    return token


def open_postgres_root(
    environ: Mapping[str, str] | None = None, *, budget: PoolBudget = DEFAULT_POOL_BUDGET
) -> PostgresRoot:
    env = os.environ if environ is None else environ
    conninfo = _required_file(env, API_POSTGRES_DSN_FILE_ENV, "PostgreSQL DSN")
    authority = _parse_authority(
        _required_file(env, API_POSTGRES_AUTHORITY_FILE_ENV, "persistence authority")
    )
    schema = env.get(API_POSTGRES_SCHEMA_ENV, "").strip() or DEFAULT_POSTGRES_SCHEMA
    try:
        database: PostgresDatabase | None = PostgresDatabase(conninfo, schema, budget)
    except (TypeError, ValueError, PostgresError):
        database = None
    del conninfo
    if database is None:
        raise PostgresRootError("PostgreSQL connection settings are invalid")
    try:
        database.start()
        started = True
    except PostgresError:
        started = False
    if not started:
        raise PostgresRootError("PostgreSQL is unreachable within the startup budget")
    try:
        database.transact(lambda connection: require_authority(connection, authority))
        verified = True
    except (AuthorityFenced, PostgresError, psycopg.Error):
        verified = False
    if not verified:
        with suppress(PostgresError):
            close_postgres_database(database)
        raise PostgresRootError(
            "PostgreSQL schema is not provisioned for this deployment's persistence authority"
        )
    return PostgresRoot(database, authority)


def close_postgres_database(
    database: PostgresDatabase, *, timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC
) -> None:
    database.stop_admission()
    database.close(timeout_sec=timeout_sec)


__all__ = [
    "API_POSTGRES_AUTHORITY_FILE_ENV",
    "API_POSTGRES_DSN_FILE_ENV",
    "API_POSTGRES_SCHEMA_ENV",
    "DEFAULT_POOL_BUDGET",
    "DEFAULT_POSTGRES_SCHEMA",
    "PostgresRoot",
    "PostgresRootError",
    "close_postgres_database",
    "open_postgres_root",
]
