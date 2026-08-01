from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from contracts.event import EventPayload
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
        payload: EventPayload = {
            "edge_event_id": edge_event_id,
            "event_type": event.event_type,
            "probability": event.probability,
            "detected_at": self.now().isoformat().replace("+00:00", "Z"),
            "camera_id": event.camera_id,
            "facility_id": event.facility_id,
            "evidence": evidence,
        }
        self.stager.stage(payload)
        clip_id = self.recorder.on_event(
            event.camera_id,
            edge_event_id,
            event.event_type,
        )
        self.stager.complete(edge_event_id, clip_id)


__all__ = ["EvidenceEventSink"]
