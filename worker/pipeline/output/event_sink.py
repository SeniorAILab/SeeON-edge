from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from contracts.event import EventPayload
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.types.business_event import BusinessEvent


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceStager(Protocol):
    def stage(self, event: EventPayload) -> None: ...

    def complete(self, edge_event_id: str, clip_id: str | None) -> None: ...


class EventClipRecorder(Protocol):
    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class EvidenceEventSink:
    """Stages an admitted event before binding its optional durable clip."""

    stager: EvidenceStager
    recorder: EventClipRecorder
    now: Callable[[], datetime] = _utc_now
    snapshot_store: SnapshotStore | None = None

    def emit(self, event: BusinessEvent) -> None:
        """Persist an event, reserve any clip, then make the event deliverable."""
        edge_event_id = str(event.identity)
        evidence: dict[str, str | int | float] = {
            "domain": event.domain,
            "identity": edge_event_id,
            "time_sec": event.time_sec,
        }
        if event.person_id is not None:
            evidence["person_id"] = event.person_id
        if event.bed_id is not None:
            evidence["bed_id"] = event.bed_id
        detected_at = self.now().isoformat().replace("+00:00", "Z")
        payload: dict[str, object] = {
            "edge_event_id": edge_event_id,
            "event_type": event.event_type,
            "probability": event.probability,
            "detected_at": detected_at,
            "camera_id": event.camera_id,
            "facility_id": event.facility_id,
            "evidence": evidence,
        }
        if event.audit is not None:
            payload["audit"] = dict(event.audit)
        if event.snapshot_jpeg is not None:
            payload["snapshot_jpeg"] = event.snapshot_jpeg
            if self.snapshot_store is not None:
                stored = self.snapshot_store.store(
                    event.snapshot_jpeg,
                    snapshot_id=edge_event_id,
                    captured_at=detected_at,
                    camera_id=event.camera_id,
                    edge_event_id=edge_event_id,
                )
                payload["snapshot"] = {
                    "snapshot_id": stored.snapshot_id,
                    "path": stored.path,
                    "sha256": stored.sha256,
                    "size_bytes": stored.size_bytes,
                    "mime_type": stored.mime_type,
                    "captured_at": stored.captured_at,
                    "camera_id": stored.camera_id,
                    "edge_event_id": stored.edge_event_id,
                }
        self.stager.stage(payload)
        clip_id = self.recorder.on_event(
            event.camera_id,
            edge_event_id,
            event.event_type,
        )
        self.stager.complete(edge_event_id, clip_id)


__all__ = ["EvidenceEventSink"]
