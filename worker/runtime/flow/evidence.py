from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from shared.events.delivery_queue import AdmissionResult
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.pipeline.diagnostics.emit_delivery import event_delivery_record
from worker.pipeline.diagnostics.record_builder import try_emit
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.flow_sealed_sidecar import (
    FlowSealedRecovery,
    FlowSealedSidecars,
)
from worker.pipeline.output.evidence.smart_record_actor import ClipSealed, SmartRecordActor
from worker.types import BusinessEvent, NativeEvidenceTrigger

LOGGER = logging.getLogger(__name__)


class FlowEvidenceStager(Protocol):
    def stage(self, event: dict[str, object]) -> AdmissionResult: ...

    def complete(self, edge_event_id: str, clip_id: str | None) -> None: ...


@dataclass(slots=True)
class FlowEvidenceBinding:
    actor: SmartRecordActor
    stager: FlowEvidenceStager
    publisher: FlowClipPublisher
    sidecars: FlowSealedSidecars
    camera_id: str
    execution_records: ExecutionRecordSink | None = None
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    _events: dict[str, BusinessEvent] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    sealed_recovery_missing_media_total: int = field(default=0, init=False)

    def emit_for_frame(self, event: BusinessEvent, trigger: NativeEvidenceTrigger) -> None:
        if event.camera_id != trigger.camera_id:
            raise ValueError("event camera does not match Flow trigger")
        detected = self.now()
        if detected.tzinfo is None or detected.utcoffset() is None:
            raise ValueError("Flow detected_at must be timezone-aware")
        detected_at = detected.isoformat().replace("+00:00", "Z")
        event_ref = str(event.identity)
        payload = {
            "edge_event_id": event_ref,
            "event_type": event.event_type,
            "probability": event.probability,
            "detected_at": detected_at,
            "camera_id": event.camera_id,
            "facility_id": event.facility_id,
            "evidence": {
                "domain": event.domain,
                "identity": event_ref,
                "time_sec": event.time_sec,
            },
        }

        def _delivery_record(admitted: bool, reason: str | None):
            return event_delivery_record(
                camera_id=event.camera_id,
                worker_boot_id=trigger.worker_boot_id,
                source_generation=trigger.source_generation,
                stream_epoch=trigger.stream_epoch,
                frame_seq=trigger.seq,
                source_pts_ns=trigger.source_pts,
                edge_event_id=event_ref,
                event_type=event.event_type,
                domain=event.domain,
                admitted=admitted,
                reason=reason,
            )

        try:
            result = self.stager.stage(payload)
        except Exception as error:
            try_emit(
                self.execution_records,
                _delivery_record(False, _admission_reason(error)),
            )
            raise
        admitted, reason = _admission_from_stage_result(result)
        if not admitted:
            try_emit(
                self.execution_records,
                _delivery_record(False, reason),
            )
            raise RuntimeError(f"event delivery admission failed: {reason}")
        try_emit(
            self.execution_records,
            _delivery_record(True, reason),
        )
        with self._lock:
            self._events[event_ref] = event
        self.actor.admit(event_ref, detected_at)

    def on_sealed(self, sealed: ClipSealed) -> None:
        with self._lock:
            events = dict(self._events)
        if sealed.contributors and all(c.event_ref not in events for c in sealed.contributors):
            return
        sidecar_path = self.sidecars.persist(sealed, events)
        recovery = FlowSealedRecovery(sealed, events, self.camera_id, sidecar_path)
        self._publish_recovery(recovery)
        with self._lock:
            for contributor in sealed.contributors:
                self._events.pop(contributor.event_ref, None)

    def replay_sealed(self) -> None:
        for recovery in self.sidecars.pending_for_camera(self.camera_id):
            media_path = Path(recovery.sealed.path)
            if not media_path.is_file():
                error = self.sidecars.discard_missing_media(recovery)
                self.sealed_recovery_missing_media_total += 1
                LOGGER.error("%s", error)
                continue
            try:
                self._publish_recovery(recovery)
            except Exception:
                LOGGER.exception(
                    "sealed Flow clip replay failed clip_id=%s camera_id=%s",
                    recovery.sealed.clip_id,
                    self.camera_id,
                )

    def _publish_recovery(self, recovery: FlowSealedRecovery) -> None:
        for contributor in recovery.sealed.contributors:
            if contributor.event_ref not in recovery.events:
                raise ValueError(
                    f"sealed Flow clip has unknown contributor {contributor.event_ref}"
                )
        published = self.publisher.publish(recovery.sealed, recovery.events)
        for contributor in recovery.sealed.contributors:
            self.stager.complete(contributor.event_ref, str(published.clip_id))
        self.sidecars.remove(recovery)


def _admission_from_stage_result(result: object) -> tuple[bool, str | None]:
    if not isinstance(result, AdmissionResult):
        return False, f"unproven-admission:{type(result).__name__}"
    reason = None if result.fault is None else str(result.fault)
    return bool(result.accepted), reason


def _admission_reason(error: BaseException) -> str:
    text = str(error)
    marker = "event delivery admission failed: "
    if text.startswith(marker):
        fault = text[len(marker) :].strip()
        if fault and fault != "None":
            return fault
    name = type(error).__name__
    return name if not text else f"{name}: {text}"


__all__ = ["FlowEvidenceBinding", "FlowEvidenceStager"]
