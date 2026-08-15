"""Read-only event, decision, and runtime projection over schema v16 facts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shared.edge_db.connection import RuntimeActor, open_runtime_database

_EDGE_EVENT_ID_MAX_LENGTH = 256


@dataclass(frozen=True, slots=True)
class EventExplanationIdentity:
    edge_event_id: str
    incident_id: str | None
    camera_id: str | None
    event_type: str | None
    detected_at: str


@dataclass(frozen=True, slots=True)
class EventExplanationDecisionValue:
    name: str
    numeric_value: float | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class EventExplanationDecision:
    decision_trace_id: str
    analysis_trace_id: str | None
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    runtime_manifest_sha256: str
    reason: str
    previous_state: str
    current_state: str
    triggered: bool
    track_id: int | None
    track_missing_reason: str | None
    bed_id: int | None
    bed_missing_reason: str | None
    values: tuple[EventExplanationDecisionValue, ...]


@dataclass(frozen=True, slots=True)
class EventExplanationRuntime:
    analysis_trace_id: str
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    frame_seq: int


@dataclass(frozen=True, slots=True)
class EventExplanationFacts:
    identity: EventExplanationIdentity
    decision: EventExplanationDecision | None
    runtime: EventExplanationRuntime | None


@dataclass(frozen=True, slots=True)
class TraceRefConflict:
    edge_event_id: str
    incident_decision_trace_id: str
    ref_decision_trace_id: str
    code: str = "TRACE_REF_CONFLICT"


class EventExplanationQuery:
    """Project one event to one identity/decision/runtime fact set."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def get(self, edge_event_id: str) -> EventExplanationFacts | TraceRefConflict | None:
        _validate_edge_event_id(edge_event_id)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(_EVENT_SELECT, (edge_event_id,)).fetchone()
            if row is None:
                return None
            identity = EventExplanationIdentity(
                edge_event_id=_required_text(row[0]),
                incident_id=_text(row[1]),
                camera_id=_text(row[2]),
                event_type=_text(row[3]),
                detected_at=_required_text(row[4]),
            )
            incident_trace_id = _text(row[5])
            ref_trace_id = _text(row[6])
            if (
                incident_trace_id is not None
                and ref_trace_id is not None
                and incident_trace_id != ref_trace_id
            ):
                return TraceRefConflict(
                    edge_event_id=identity.edge_event_id,
                    incident_decision_trace_id=incident_trace_id,
                    ref_decision_trace_id=ref_trace_id,
                )
            resolved_trace_id = (
                incident_trace_id if incident_trace_id is not None else ref_trace_id
            )
            if resolved_trace_id is None:
                return EventExplanationFacts(identity=identity, decision=None, runtime=None)
            return EventExplanationFacts(
                identity=identity,
                decision=_load_decision(connection, resolved_trace_id),
                runtime=_load_runtime(connection, resolved_trace_id),
            )
        finally:
            connection.close()


def _load_decision(
    connection: sqlite3.Connection,
    decision_trace_id: str,
) -> EventExplanationDecision | None:
    row = connection.execute(_DECISION_SELECT, (decision_trace_id,)).fetchone()
    if row is None:
        return None
    values = connection.execute(_VALUES_SELECT, (decision_trace_id,)).fetchall()
    return EventExplanationDecision(
        decision_trace_id=_required_text(row[0]),
        analysis_trace_id=_text(row[1]),
        module_qualified_id=_required_text(row[2]),
        policy_qualified_id=_required_text(row[3]),
        effective_policy_id=_required_text(row[4]),
        runtime_manifest_sha256=_required_text(row[5]),
        reason=_required_text(row[6]),
        previous_state=_required_text(row[7]),
        current_state=_required_text(row[8]),
        triggered=_required_flag(row[9]),
        track_id=_optional_int(row[10]),
        track_missing_reason=_text(row[11]),
        bed_id=_optional_int(row[12]),
        bed_missing_reason=_text(row[13]),
        values=tuple(
            EventExplanationDecisionValue(
                name=_required_text(value[0]),
                numeric_value=_optional_float(value[1]),
                missing_reason=_text(value[2]),
            )
            for value in values
        ),
    )


def _load_runtime(
    connection: sqlite3.Connection,
    decision_trace_id: str,
) -> EventExplanationRuntime | None:
    row = connection.execute(_RUNTIME_SELECT, (decision_trace_id,)).fetchone()
    if row is None:
        return None
    return EventExplanationRuntime(
        analysis_trace_id=_required_text(row[0]),
        worker_boot_id=_required_text(row[1]),
        camera_id=_required_text(row[2]),
        stream_epoch=_required_int(row[3]),
        frame_seq=_required_int(row[4]),
    )


def _validate_edge_event_id(edge_event_id: str) -> None:
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


def _required_flag(value: object) -> bool:
    if value not in (0, 1):
        raise TypeError("stored flag is invalid")
    return bool(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else _required_int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("stored numeric value is invalid")
    return float(value)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


_EVENT_SELECT = """
SELECT event.edge_event_id, incident.incident_id, incident.camera_id,
       incident.event_type, event.detected_at, incident.decision_trace_id,
       ref.decision_trace_id
FROM evidence_events AS event
LEFT JOIN evidence_incidents AS incident
  ON incident.edge_event_id = event.edge_event_id
LEFT JOIN evidence_event_trace_refs AS ref
  ON ref.edge_event_id = event.edge_event_id
WHERE event.edge_event_id = ?
"""

_DECISION_SELECT = """
SELECT trace.trace_id, trace.analysis_trace_id, trace.module_qualified_id,
       trace.policy_qualified_id, trace.effective_policy_id,
       trace.runtime_manifest_sha256, trace.reason, trace.previous_state,
       trace.current_state, trace.triggered, trace.track_id,
       trace.track_missing_reason, trace.bed_id, trace.bed_missing_reason
FROM evidence_decision_traces AS trace
WHERE trace.trace_id = ?
"""

_VALUES_SELECT = """
SELECT name, numeric_value, missing_reason
FROM evidence_decision_values
WHERE decision_trace_id = ?
ORDER BY name
"""

_RUNTIME_SELECT = """
SELECT analysis.trace_id, analysis.worker_boot_id, analysis.camera_id,
       analysis.stream_epoch, analysis.frame_seq
FROM evidence_decision_traces AS trace
JOIN runtime_analysis_traces AS analysis
  ON analysis.trace_id = trace.analysis_trace_id
WHERE trace.trace_id = ?
"""

__all__ = [
    "EventExplanationDecision",
    "EventExplanationDecisionValue",
    "EventExplanationFacts",
    "EventExplanationIdentity",
    "EventExplanationQuery",
    "EventExplanationRuntime",
    "TraceRefConflict",
]
