from __future__ import annotations

from typing import NotRequired, TypeAlias, TypedDict

from contracts.event import EventEvidence, EventScalar


class WorkerEventPayload(TypedDict):
    edge_event_id: NotRequired[str]
    event_type: NotRequired[str]
    probability: NotRequired[float]
    confidence: NotRequired[float]
    detected_at: NotRequired[str]
    camera_id: NotRequired[str]
    facility_id: NotRequired[str]
    domain: NotRequired[str]
    identity: NotRequired[str | int]
    time_sec: NotRequired[float]
    person_id: NotRequired[int | None]
    bed_id: NotRequired[int | None]
    idempotency_key: NotRequired[str]
    event_id: NotRequired[str]
    clip_id: NotRequired[str]
    evidence: NotRequired[EventEvidence]
    audit: NotRequired[EventEvidence]
    snapshot_jpeg: NotRequired[bytes]
    snapshot: NotRequired[EventEvidence]


MutableWorkerEventPayload: TypeAlias = dict[str, EventScalar | bytes | dict[str, EventScalar]]


__all__ = ["MutableWorkerEventPayload", "WorkerEventPayload"]
