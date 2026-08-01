"""Inference deadline watchdog (todo 26).

A GPU forward pass that never returns cannot be interrupted from Python: the
calling thread is blocked inside the driver.  This watchdog therefore runs on an
independent thread, notices that an in-flight inference has passed its deadline,
and drives the *same* fault boundary a direct CUDA fault would drive —
``FaultHandler.handle()``, which persists exactly one first-fault record, stops
every camera loop, and hard-exits with
:data:`~worker.runtime.faults.handler.FATAL_ACCELERATOR_EXIT_CODE` (4).  The exit
code is shared deliberately: from an operator's point of view a hung forward pass
and a failed forward pass are the same condition — the accelerator context is
unusable and must not be reused in-process.

**Scope, stated plainly.** This is best-effort application containment, not a
promise to recover a hung driver.  Exiting the process releases what a process
can release; a wedged GPU or an Xid event remains host-runbook territory and the
worker never queries or claims Xid state.

The monitor loop is a thin wrapper over :meth:`InferenceWatchdog.check`, so the
expiry decision is testable with an injected clock and no real waiting, while a
subprocess test covers the real thread plus the real ``os._exit``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from time import monotonic
from typing import Final, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE, FaultHandler
from worker.runtime.faults.record import make_fault_record

LOGGER: Final = logging.getLogger(__name__)

WATCHDOG_STAGE: Final = "inference_watchdog"
DEFAULT_INFERENCE_DEADLINE_SEC: Final = 30.0
# Poll at a quarter of the deadline so expiry is detected promptly without
# spinning, and never slower than once a second.
_MAX_POLL_INTERVAL_SEC: Final = 1.0

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class InFlightInference:
    """One registered in-flight forward pass and its absolute deadline."""

    token: int
    camera_id: str
    task: str
    frame_index: int | None
    started_at: float
    deadline_at: float
    model_artifact_digest: str | None


@final
class InferenceWatchdog:
    """Fire the shared fault boundary when a forward pass overruns its deadline.

    Parameters
    ----------
    handler:
        The composition root's :class:`FaultHandler`.  The watchdog does not own
        a second exit path; it drives the one the root already registered, so a
        watchdog trip and a direct CUDA fault are indistinguishable downstream.
    profile:
        Active profile name, recorded in the first-fault record.
    deadline_sec:
        Default deadline for a single forward pass.  Per-call overrides exist
        because a pose forward and an LSTM window forward have different costs.
    clock:
        Injected monotonic clock.  Must be monotonic, never wall time.
    """

    def __init__(
        self,
        handler: FaultHandler,
        *,
        profile: str,
        deadline_sec: float = DEFAULT_INFERENCE_DEADLINE_SEC,
        clock: Clock = monotonic,
    ) -> None:
        if deadline_sec <= 0:
            message = f"inference deadline must be positive, received {deadline_sec}"
            raise ValueError(message)
        self._handler = handler
        self._profile = profile
        self._deadline_sec = deadline_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._in_flight: dict[int, InFlightInference] = {}
        self._tokens = count()
        self._tripped = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def deadline_sec(self) -> float:
        return self._deadline_sec

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    def in_flight(self) -> tuple[InFlightInference, ...]:
        """Immutable snapshot of currently registered forward passes."""
        with self._lock:
            return tuple(self._in_flight.values())

    def register(
        self,
        *,
        camera_id: str,
        task: str,
        frame_index: int | None = None,
        deadline_sec: float | None = None,
        model_artifact_digest: str | None = None,
    ) -> int:
        """Record a forward pass as in-flight and return its completion token."""
        now = self._clock()
        budget = self._deadline_sec if deadline_sec is None else deadline_sec
        with self._lock:
            token = next(self._tokens)
            self._in_flight[token] = InFlightInference(
                token=token,
                camera_id=camera_id,
                task=task,
                frame_index=frame_index,
                started_at=now,
                deadline_at=now + budget,
                model_artifact_digest=model_artifact_digest,
            )
        return token

    def complete(self, token: int) -> None:
        """Clear a forward pass.  Unknown or already-cleared tokens are ignored."""
        with self._lock:
            _ = self._in_flight.pop(token, None)

    @contextmanager
    def guard(
        self,
        *,
        camera_id: str,
        task: str,
        frame_index: int | None = None,
        deadline_sec: float | None = None,
        model_artifact_digest: str | None = None,
    ) -> Iterator[int]:
        """Guard one forward pass; the token is cleared even if inference raises."""
        token = self.register(
            camera_id=camera_id,
            task=task,
            frame_index=frame_index,
            deadline_sec=deadline_sec,
            model_artifact_digest=model_artifact_digest,
        )
        try:
            yield token
        finally:
            self.complete(token)

    def check(self) -> InFlightInference | None:
        """Fire the fault boundary for the oldest overdue pass, if any.

        Returns the entry that tripped, or ``None``.  Trips at most once per
        process: after the first trip the process is exiting, and a second record
        would describe a consequence rather than the cause.
        """
        overdue = self._claim_overdue()
        if overdue is None:
            return None
        self._trip(overdue)
        return overdue

    def start(self) -> None:
        """Start the monitor thread.  Idempotent."""
        if self._thread is not None:
            return
        self._stopped.clear()
        thread = threading.Thread(
            target=self._monitor,
            name="inference-watchdog",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the monitor thread on clean shutdown.  Idempotent."""
        self._stopped.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def __enter__(self) -> InferenceWatchdog:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _claim_overdue(self) -> InFlightInference | None:
        """Atomically claim the one trip slot for the oldest overdue pass.

        The trip flag is set inside the same lock that reads the in-flight map, so
        the monitor thread and a caller-driven ``check()`` cannot both trip on the
        same overrun and race to persist two records.
        """
        now = self._clock()
        with self._lock:
            if self._tripped.is_set():
                return None
            overdue = [entry for entry in self._in_flight.values() if entry.deadline_at < now]
            if not overdue:
                return None
            self._tripped.set()
            return min(overdue, key=lambda entry: (entry.started_at, entry.token))

    def _trip(self, entry: InFlightInference) -> None:
        elapsed = self._clock() - entry.started_at
        message = (
            f"inference deadline exceeded: task {entry.task!r} on camera "
            f"{entry.camera_id!r} did not complete within "
            f"{entry.deadline_at - entry.started_at:.3f}s (elapsed {elapsed:.3f}s)"
        )
        LOGGER.critical(
            "%s; retiring the accelerator context and exiting %d",
            message,
            FATAL_ACCELERATOR_EXIT_CODE,
        )
        error = FatalAcceleratorError(message, camera_id=entry.camera_id, task=entry.task)
        record = make_fault_record(
            error,
            profile=self._profile,
            task=entry.task,
            stage=WATCHDOG_STAGE,
            camera_id=entry.camera_id,
            frame_index=entry.frame_index,
            model_artifact_digest=entry.model_artifact_digest,
            exit_code=FATAL_ACCELERATOR_EXIT_CODE,
        )
        self._handler.handle(error, record)

    def _poll_interval(self) -> float:
        return min(self._deadline_sec / 4.0, _MAX_POLL_INTERVAL_SEC)

    def _monitor(self) -> None:
        interval = self._poll_interval()
        while not self._stopped.wait(interval):
            if self.check() is not None:
                return


__all__ = [
    "DEFAULT_INFERENCE_DEADLINE_SEC",
    "WATCHDOG_STAGE",
    "Clock",
    "InFlightInference",
    "InferenceWatchdog",
]
