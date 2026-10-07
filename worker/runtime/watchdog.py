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
_MAX_POLL_INTERVAL_SEC: Final = 1.0

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class InFlightInference:
    token: int
    camera_id: str
    task: str
    frame_index: int | None
    started_at: float
    deadline_at: float
    model_artifact_digest: str | None


@final
class InferenceWatchdog:
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
        overdue = self._claim_overdue()
        if overdue is None:
            return None
        self._trip(overdue)
        return overdue

    def start(self) -> None:
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
