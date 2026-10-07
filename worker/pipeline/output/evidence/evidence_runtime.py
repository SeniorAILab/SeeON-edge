from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

from shared.boundary import ProbeFailure, probe
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
    SenderStep,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

LOGGER: Final = logging.getLogger(__name__)
SENDER_FAILURE_LOG_INTERVAL_SECONDS: Final = 300.0


class SenderProtocol(Protocol):
    def run_once(self) -> SenderStep: ...


@dataclass(slots=True)
class EvidenceExportRuntime:
    store_dir: Path
    queue_directory: Path
    sender: SenderProtocol
    on_open: Callable[[], None] | None = None
    on_reconciled: Callable[[], None] | None = None
    _initialized: bool = field(default=False, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _wake_sender: threading.Event = field(default_factory=threading.Event, init=False)
    _lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    monotonic: Callable[[], float] = field(default=time.monotonic, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _last_failure_log_at: float | None = field(default=None, init=False)

    @classmethod
    def from_config(
        cls,
        *,
        store_dir: Path,
        queue_directory: Path,
        relay_url: str,
        relay_token: str | None,
        probe_camera_id: str,
        clip_export_enabled: Callable[[], bool],
        flow_sealed_sidecar_directory: Path | None = None,
        execution_records: ExecutionRecordSink | None = None,
        observing_boot_id: str | None = None,
    ) -> EvidenceExportRuntime:
        token = "" if relay_token is None else relay_token.strip()
        if not relay_url.strip() or not token or not probe_camera_id.strip():
            raise ValueError("evidence delivery requires relay URL, token, and camera identity")
        config = SenderConfig(
            relay_url=relay_url, relay_token=token, probe_camera_id=probe_camera_id
        )
        return cls(
            store_dir=store_dir,
            queue_directory=queue_directory,
            sender=EvidenceSender(
                queue_directory,
                config,
                clip_export_enabled=clip_export_enabled,
                flow_sealed_sidecar_directory=flow_sealed_sidecar_directory,
                execution_records=execution_records,
                observing_boot_id=observing_boot_id,
            ),
        )

    def initialize_under_lock(self) -> None:
        self._initialized = False
        if self.on_open is not None:
            self.on_open()
        if self.on_reconciled is not None:
            self.on_reconciled()
        self._initialized = True

    def stager(
        self,
        *,
        camera_id: str,
        facility_id: str,
        resident_id: str | None,
        config_version: int,
    ) -> DurableEvidenceStager:
        return DurableEvidenceStager(
            self.queue_directory,
            camera_id=camera_id,
            facility_id=facility_id,
            resident_id=resident_id,
            config_version=config_version,
            clock=time.time,
        )

    def notify_clip_finalized(self, clip_id: str) -> None:
        del clip_id
        self._wake_sender.set()

    def start_sender(self) -> None:
        with self._lifecycle_lock:
            if not self._initialized:
                raise RuntimeError("evidence runtime must initialize before sender start")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run_sender,
                name="evidence-sender",
                daemon=True,
            )
            thread.start()
            self._thread = thread

    def stop_sender(self, *, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            self._stop_event.set()
            self._wake_sender.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _note_sender_failure(self, failure: ProbeFailure) -> None:
        self._consecutive_failures += 1
        now = self.monotonic()
        last = self._last_failure_log_at
        if last is not None and now - last < SENDER_FAILURE_LOG_INTERVAL_SECONDS:
            return
        self._last_failure_log_at = now
        if self._consecutive_failures == 1 and failure.error is not None:
            LOGGER.debug("evidence sender tick first failure traceback", exc_info=failure.error)
        LOGGER.warning(
            "evidence sender tick failing exception_class=%s failures=%d",
            failure.reason.partition(":")[0],
            self._consecutive_failures,
        )

    def _note_sender_success(self) -> None:
        if self._consecutive_failures:
            LOGGER.info("evidence sender recovered failures=%d", self._consecutive_failures)
        self._consecutive_failures = 0
        self._last_failure_log_at = None

    def _run_sender(self) -> None:
        while not self._stop_event.is_set():
            outcome = probe(self.sender.run_once)
            if isinstance(outcome, ProbeFailure):
                step = SenderStep.RETRY_SCHEDULED
                self._note_sender_failure(outcome)
            else:
                step = outcome
                self._note_sender_success()
            if step not in {SenderStep.EVENT_ACKED, SenderStep.CLIP_ACKED}:
                self._wake_sender.wait(1.0)
                self._wake_sender.clear()


__all__ = ["EvidenceExportRuntime"]
