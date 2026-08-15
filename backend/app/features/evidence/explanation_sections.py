"""Read-only delivery, media, and correlation projection over schema v16 facts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.evidence.explanation_schemas import (
    AlertIdFact,
    AttemptCountFact,
    BackendEventIdFact,
    DeliveryDispositionFact,
    EventExplanationArtifact,
    EventExplanationCorrelation,
    EventExplanationDelivery,
    EventExplanationMedia,
    HttpStatusFact,
    OutboxStateFact,
)
from shared.edge_db.connection import RuntimeActor, open_runtime_database

_EDGE_EVENT_ID_MAX_LENGTH = 256


class AlertCorrelationExport(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    edge_event_id: str = Field(min_length=1)
    alert_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class EventExplanationSections:
    delivery: EventExplanationDelivery
    media: EventExplanationMedia
    correlation: EventExplanationCorrelation


def project_explanation_sections(
    database_path: Path,
    edge_event_id: str,
    *,
    alert_correlation_export: AlertCorrelationExport | None = None,
) -> EventExplanationSections:
    _require_edge_event_id(edge_event_id)
    connection = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        row = connection.execute(_SECTION_SELECT, (edge_event_id,)).fetchone()
    finally:
        connection.close()
    if row is None:
        return EventExplanationSections(
            delivery=_unresolved_delivery(),
            media=_unresolved_media(),
            correlation=_unavailable_correlation(),
        )
    return EventExplanationSections(
        delivery=_project_delivery(row),
        media=_project_media(row),
        correlation=_project_correlation(edge_event_id, alert_correlation_export),
    )


def _project_delivery(row: sqlite3.Row | tuple[object, ...]) -> EventExplanationDelivery:
    outbox_state = _required_text(row[1])
    delivery_state = _required_text(row[2])
    attempt_count = _required_int(row[3])
    backend_event_id = _text(row[4])
    projected_state = _projected_outbox_state(outbox_state, delivery_state)
    disposition = _exact_disposition(delivery_state, attempt_count)
    backend = (
        BackendEventIdFact(value=backend_event_id)
        if backend_event_id is not None
        else BackendEventIdFact(missing_reason="backend_event_id_not_persisted")
    )
    outbox_fact = (
        OutboxStateFact(value=projected_state)
        if projected_state is not None
        else OutboxStateFact(missing_reason="delivery_never_attempted")
    )
    return EventExplanationDelivery(
        status="COMPLETE",
        reasons=[],
        outbox_state=outbox_fact,
        attempt_count=AttemptCountFact(value=attempt_count),
        last_delivery_disposition=disposition,
        last_http_status=HttpStatusFact(missing_reason="last_http_status_not_persisted"),
        backend_event_id=backend,
    )


def _projected_outbox_state(outbox_state: str, delivery_state: str) -> str | None:
    if delivery_state in {"ACKED", "PERMANENT", "COMPATIBILITY"}:
        return delivery_state
    if outbox_state == "ACKED":
        return "ACKED"
    if outbox_state == "STAGED":
        return None
    return "PENDING"


def _exact_disposition(delivery_state: str, attempt_count: int) -> DeliveryDispositionFact:
    if delivery_state in {"PERMANENT", "COMPATIBILITY"}:
        return DeliveryDispositionFact(value=delivery_state)
    if attempt_count == 0:
        return DeliveryDispositionFact(missing_reason="delivery_never_attempted")
    return DeliveryDispositionFact(missing_reason="disposition_not_persisted")


def _project_media(row: sqlite3.Row | tuple[object, ...]) -> EventExplanationMedia:
    snapshot_state = _text(row[5])
    snapshot_reason = _text(row[6])
    clip_id = _text(row[7])
    clip_slot_state = _text(row[8])
    clip_slot_reason = _text(row[9])
    retention_state = _text(row[10])
    clip_local_state = _text(row[11])
    clip_publish_state = _text(row[12])
    derivative_state = _text(row[13])
    snapshot = _artifact(
        snapshot_state,
        recorded_reason=snapshot_reason,
        absent_reason="snapshot_not_recorded",
    )
    clip = _clip_artifact(
        clip_id=clip_id,
        slot_state=clip_slot_state,
        slot_reason=clip_slot_reason,
        retention_state=retention_state,
        local_state=clip_local_state,
        publish_state=clip_publish_state,
    )
    reasons: list[str] = []
    if snapshot.state != "AVAILABLE" and snapshot.missing_reason is not None:
        reasons.append(snapshot.missing_reason)
    if clip.state != "AVAILABLE" and clip.missing_reason is not None:
        reasons.append(clip.missing_reason)
    if derivative_state in {"UNAVAILABLE", "CORRUPT"}:
        reasons.append("artifact_unavailable")
    unique_reasons = list(dict.fromkeys(reasons))
    if snapshot.state == "AVAILABLE" and clip.state == "AVAILABLE" and not unique_reasons:
        return EventExplanationMedia(
            status="COMPLETE",
            reasons=[],
            snapshot=snapshot,
            clip=clip,
        )
    in_progress = snapshot.state == "PENDING" or clip.state == "PENDING"
    known = snapshot.state == "AVAILABLE" or clip.state == "AVAILABLE" or in_progress
    return EventExplanationMedia(
        status="PARTIAL" if known else "UNAVAILABLE",
        reasons=unique_reasons,
        snapshot=snapshot,
        clip=clip,
    )


def _artifact(
    state: str | None,
    *,
    recorded_reason: str | None,
    absent_reason: str,
) -> EventExplanationArtifact:
    del recorded_reason
    if state == "AVAILABLE":
        return EventExplanationArtifact(state="AVAILABLE")
    if state == "PENDING":
        return EventExplanationArtifact(state="PENDING")
    if state == "CORRUPT":
        return EventExplanationArtifact(state="CORRUPT", missing_reason="artifact_unavailable")
    if state == "UNAVAILABLE":
        return EventExplanationArtifact(state="UNAVAILABLE", missing_reason=absent_reason)
    return EventExplanationArtifact(state="NOT_RECORDED", missing_reason=absent_reason)


def _clip_artifact(
    *,
    clip_id: str | None,
    slot_state: str | None,
    slot_reason: str | None,
    retention_state: str | None,
    local_state: str | None,
    publish_state: str | None,
) -> EventExplanationArtifact:
    del slot_reason
    if clip_id is None:
        return EventExplanationArtifact(state="NOT_RECORDED", missing_reason="clip_not_recorded")
    if retention_state == "PURGED" or slot_state == "UNAVAILABLE":
        return EventExplanationArtifact(state="UNAVAILABLE", missing_reason="artifact_unavailable")
    if slot_state == "CORRUPT" or local_state == "CORRUPT":
        return EventExplanationArtifact(state="CORRUPT", missing_reason="artifact_unavailable")
    if (
        local_state == "VERIFIED"
        and publish_state == "PUBLISHED"
        and slot_state == "AVAILABLE"
    ):
        return EventExplanationArtifact(state="AVAILABLE")
    return EventExplanationArtifact(state="PENDING")


def _project_correlation(
    edge_event_id: str,
    export: AlertCorrelationExport | None,
) -> EventExplanationCorrelation:
    if export is None or export.edge_event_id != edge_event_id:
        return _unavailable_correlation()
    return EventExplanationCorrelation(
        status="COMPLETE",
        reasons=[],
        alert_id=AlertIdFact(value=export.alert_id),
    )


def _unavailable_correlation() -> EventExplanationCorrelation:
    return EventExplanationCorrelation(
        status="UNAVAILABLE",
        reasons=["alert_correlation_export_not_supplied"],
        alert_id=AlertIdFact(missing_reason="alert_correlation_export_not_supplied"),
    )


def _unresolved_delivery() -> EventExplanationDelivery:
    return EventExplanationDelivery(
        status="UNAVAILABLE",
        reasons=["outbox_row_unresolved"],
        outbox_state=OutboxStateFact(missing_reason="outbox_row_unresolved"),
        attempt_count=AttemptCountFact(missing_reason="outbox_row_unresolved"),
        last_delivery_disposition=DeliveryDispositionFact(
            missing_reason="outbox_row_unresolved"
        ),
        last_http_status=HttpStatusFact(missing_reason="last_http_status_not_persisted"),
        backend_event_id=BackendEventIdFact(missing_reason="outbox_row_unresolved"),
    )


def _unresolved_media() -> EventExplanationMedia:
    return EventExplanationMedia(
        status="UNAVAILABLE",
        reasons=["clip_not_recorded"],
        snapshot=EventExplanationArtifact(
            state="NOT_RECORDED",
            missing_reason="snapshot_not_recorded",
        ),
        clip=EventExplanationArtifact(state="NOT_RECORDED", missing_reason="clip_not_recorded"),
    )


def _require_edge_event_id(edge_event_id: str) -> None:
    if (
        not isinstance(edge_event_id, str)
        or not edge_event_id
        or len(edge_event_id) > _EDGE_EVENT_ID_MAX_LENGTH
        or "\x00" in edge_event_id
    ):
        raise ValueError("invalid edge_event_id")


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("stored text is invalid")
    return value


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored integer is invalid")
    return value


def _text(value: object) -> str | None:
    return None if value is None else str(value)


_SECTION_SELECT = """
SELECT event.edge_event_id, event.state, event.delivery_state, event.attempt_count,
       event.backend_event_id, snapshot_slot.state, snapshot_slot.reason,
       incident.primary_clip_id, clip_slot.state, clip_slot.reason, retention.state,
       clip.local_state, clip.publish_state, derivative.state
FROM evidence_events AS event
LEFT JOIN evidence_incidents AS incident
  ON incident.edge_event_id = event.edge_event_id
LEFT JOIN evidence_artifact_slots AS snapshot_slot
  ON snapshot_slot.incident_id = incident.incident_id
 AND snapshot_slot.slot_name = 'SNAPSHOT'
LEFT JOIN evidence_artifact_slots AS clip_slot
  ON clip_slot.incident_id = incident.incident_id
 AND clip_slot.slot_name = 'PRIMARY_CLIP'
LEFT JOIN evidence_retention_states AS retention
  ON retention.clip_id = incident.primary_clip_id
LEFT JOIN evidence_clips AS clip
  ON clip.clip_id = incident.primary_clip_id
LEFT JOIN derivative_evidence_slots AS derivative
  ON derivative.incident_id = incident.incident_id
 AND derivative.derivative_kind = 'ANNOTATED_CLIP'
WHERE event.edge_event_id = ?
"""


__all__ = [
    "AlertCorrelationExport",
    "EventExplanationSections",
    "project_explanation_sections",
]
