"""Allowlisted attribution evidence over the current false-positive cohort."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shared.edge_db.event_neighborhood import (
    EXPECTED_NEIGHBORHOOD_FRAMES,
    NeighborhoodCoverage,
    NeighborhoodTrigger,
    coverage_for_decision,
)
from worker.fp_attribution.cohort import (
    FalsePositiveCohortExclusion,
    FalsePositiveCohortMember,
    FalsePositiveCohortQuery,
    open_query_only_connection,
)
from worker.types.trace import DecisionTraceReason, DecisionTraceState

_SCORE_NAME = "fall_probability"
_THRESHOLD_NAME = "operating_threshold"
_VALUE_ABSENT = "value_not_persisted"
EvidenceStatus = Literal["COMPLETE", "PRUNED", "UNKNOWN"]


@dataclass(frozen=True, slots=True)
class AttributionEvidenceRecord:
    edge_event_id: str
    decision_reason: str | None
    previous_state: str | None
    current_state: str | None
    score: float | None
    threshold: float | None
    score_missing_reason: str | None
    threshold_missing_reason: str | None
    track_id: int | None
    track_missing_reason: str | None
    track_changed: bool
    bed_id: int | None
    bed_missing_reason: str | None
    bed_changed: bool
    worker_boot_id: str | None
    stream_epoch: int | None
    boot_changed: bool
    epoch_changed: bool
    associated_sibling_event_ids: tuple[str, ...]
    attempt_count: int
    backend_event_ids: tuple[str, ...]
    coverage_status: str
    coverage_reason: str | None
    expected_frames: int
    retained_frames: int
    neighborhood_pruned: bool
    evidence_status: EvidenceStatus
    category: None
    prevented_eligible: bool


@dataclass(frozen=True, slots=True)
class AttributionEvidence:
    records: tuple[AttributionEvidenceRecord, ...]
    exclusions: tuple[FalsePositiveCohortExclusion, ...]


class AttributionEvidenceQuery:
    """Compose one allowlisted evidence record per current cohort member."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def extract(self) -> AttributionEvidence:
        cohort = FalsePositiveCohortQuery(self.database_path).load()
        connection = open_query_only_connection(self.database_path)
        try:
            records = tuple(_record_for(connection, member) for member in cohort.members)
        finally:
            connection.close()
        return AttributionEvidence(records=records, exclusions=cohort.exclusions)


def _record_for(
    connection: sqlite3.Connection,
    member: FalsePositiveCohortMember,
) -> AttributionEvidenceRecord:
    coverage = _coverage(connection, member.decision_trace_id)
    decision = _load_decision(connection, member.decision_trace_id)
    score, score_reason, threshold, threshold_reason = _load_values(
        connection,
        member.decision_trace_id,
    )
    attempt_count, backend_ids = _load_delivery(connection, member.edge_event_id)
    siblings = _load_siblings(connection, member.edge_event_id)
    track_id = None if decision is None else decision[3]
    bed_id = None if decision is None else decision[5]
    trigger = coverage.trigger
    track_changed, bed_changed = _change_facts(connection, trigger, track_id, bed_id)
    reason = None if decision is None else _closed_reason(decision[0])
    previous_state = None if decision is None else _closed_state(decision[1])
    current_state = None if decision is None else _closed_state(decision[2])
    decision_tokens_trusted = decision is None or (
        reason is not None and previous_state is not None and current_state is not None
    )
    if coverage.neighborhood_pruned:
        status: EvidenceStatus = "PRUNED"
        eligible = False
    elif score is None or threshold is None or not decision_tokens_trusted:
        status = "UNKNOWN"
        eligible = False
    else:
        status = "COMPLETE"
        eligible = coverage.prevented_eligible
    return AttributionEvidenceRecord(
        edge_event_id=member.edge_event_id,
        decision_reason=reason,
        previous_state=previous_state,
        current_state=current_state,
        score=score,
        threshold=threshold,
        score_missing_reason=score_reason,
        threshold_missing_reason=threshold_reason,
        track_id=track_id,
        track_missing_reason=None if decision is None else decision[4],
        track_changed=track_changed,
        bed_id=bed_id,
        bed_missing_reason=None if decision is None else decision[6],
        bed_changed=bed_changed,
        worker_boot_id=None if trigger is None else trigger.worker_boot_id,
        stream_epoch=None if trigger is None else trigger.stream_epoch,
        boot_changed=False,
        epoch_changed=False,
        associated_sibling_event_ids=siblings,
        attempt_count=attempt_count,
        backend_event_ids=backend_ids,
        coverage_status=coverage.status,
        coverage_reason=coverage.coverage_reason,
        expected_frames=coverage.expected_frames,
        retained_frames=coverage.retained_frames,
        neighborhood_pruned=coverage.neighborhood_pruned,
        evidence_status=status,
        category=None,
        prevented_eligible=eligible,
    )


def _coverage(
    connection: sqlite3.Connection,
    decision_trace_id: str | None,
) -> NeighborhoodCoverage:
    if decision_trace_id is None:
        return coverage_for_decision(connection, "")
    return coverage_for_decision(connection, decision_trace_id)


def _load_decision(
    connection: sqlite3.Connection,
    decision_trace_id: str | None,
) -> tuple[str, str, str, int | None, str | None, int | None, str | None] | None:
    if decision_trace_id is None:
        return None
    row = connection.execute(
        """
        SELECT reason, previous_state, current_state, track_id, track_missing_reason,
               bed_id, bed_missing_reason
        FROM evidence_decision_traces
        WHERE trace_id = ?
        """,
        (decision_trace_id,),
    ).fetchone()
    if row is None:
        return None
    return (
        _required_text(row[0]),
        _required_text(row[1]),
        _required_text(row[2]),
        _optional_int(row[3]),
        _text(row[4]),
        _optional_int(row[5]),
        _text(row[6]),
    )


def _load_values(
    connection: sqlite3.Connection,
    decision_trace_id: str | None,
) -> tuple[float | None, str | None, float | None, str | None]:
    score: float | None = None
    score_reason: str | None = _VALUE_ABSENT
    threshold: float | None = None
    threshold_reason: str | None = _VALUE_ABSENT
    if decision_trace_id is None:
        return score, score_reason, threshold, threshold_reason
    rows = connection.execute(
        """
        SELECT name, numeric_value, missing_reason
        FROM evidence_decision_values
        WHERE decision_trace_id = ?
        """,
        (decision_trace_id,),
    ).fetchall()
    for row in rows:
        name = _required_text(row[0])
        numeric = _optional_float(row[1])
        stored_reason = _text(row[2])
        if name == _SCORE_NAME:
            score, score_reason = _named_value(numeric, stored_reason)
        elif name == _THRESHOLD_NAME:
            threshold, threshold_reason = _named_value(numeric, stored_reason)
    return score, score_reason, threshold, threshold_reason


def _named_value(
    numeric: float | None,
    stored_reason: str | None,
) -> tuple[float | None, str | None]:
    if numeric is not None:
        return numeric, None
    return None, stored_reason or _VALUE_ABSENT


def _load_delivery(
    connection: sqlite3.Connection,
    edge_event_id: str,
) -> tuple[int, tuple[str, ...]]:
    row = connection.execute(
        """
        SELECT attempt_count, backend_event_id
        FROM evidence_events
        WHERE edge_event_id = ?
        """,
        (edge_event_id,),
    ).fetchone()
    if row is None:
        return 0, ()
    backend = _text(row[1])
    return _required_int(row[0]), () if backend is None else (backend,)


def _load_siblings(connection: sqlite3.Connection, edge_event_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT other.edge_event_id
        FROM clip_events AS mine
        JOIN clip_events AS other
          ON other.clip_id = mine.clip_id
         AND other.edge_event_id != mine.edge_event_id
        WHERE mine.edge_event_id = ?
        ORDER BY other.edge_event_id
        """,
        (edge_event_id,),
    ).fetchall()
    return tuple(_required_text(row[0]) for row in rows)


def _change_facts(
    connection: sqlite3.Connection,
    trigger: NeighborhoodTrigger | None,
    decision_track_id: int | None,
    decision_bed_id: int | None,
) -> tuple[bool, bool]:
    if trigger is None:
        return False, False
    window_start = trigger.frame_seq - (EXPECTED_NEIGHBORHOOD_FRAMES - 1)
    if window_start < 0:
        window_start = 0
    identity = (
        trigger.worker_boot_id,
        trigger.camera_id,
        trigger.stream_epoch,
        window_start,
        trigger.frame_seq,
    )
    track_rows = connection.execute(
        """
        SELECT persons.track_id
        FROM runtime_analysis_traces AS analysis
        JOIN runtime_analysis_persons AS persons
          ON persons.analysis_trace_id = analysis.trace_id
        WHERE analysis.worker_boot_id = ?
          AND analysis.camera_id = ?
          AND analysis.stream_epoch = ?
          AND analysis.frame_seq >= ?
          AND analysis.frame_seq <= ?
        """,
        identity,
    ).fetchall()
    seen_tracks = {
        value for value in (_optional_int(row[0]) for row in track_rows) if value is not None
    }
    if not seen_tracks:
        track_changed = False
    elif decision_track_id is None:
        track_changed = len(seen_tracks) > 1
    else:
        track_changed = seen_tracks != {decision_track_id}
    bed_rows = connection.execute(
        """
        SELECT analysis.frame_seq, COUNT(beds.ordinal)
        FROM runtime_analysis_traces AS analysis
        LEFT JOIN runtime_analysis_beds AS beds
          ON beds.analysis_trace_id = analysis.trace_id
        WHERE analysis.worker_boot_id = ?
          AND analysis.camera_id = ?
          AND analysis.stream_epoch = ?
          AND analysis.frame_seq >= ?
          AND analysis.frame_seq <= ?
        GROUP BY analysis.frame_seq
        """,
        identity,
    ).fetchall()
    bed_counts = {int(row[1]) for row in bed_rows}
    bed_changed = len(bed_counts) > 1 or (
        decision_bed_id is not None and any(count == 0 for count in bed_counts)
    )
    return track_changed, bed_changed


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("stored text is invalid")
    return value


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored integer is invalid")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else _required_int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("stored numeric value is invalid")
    return float(value)


def _text(value: object) -> str | None:
    return None if value is None else _required_text(value)


_CLOSED_REASONS = frozenset(item.value for item in DecisionTraceReason)
_CLOSED_STATES = frozenset(item.value for item in DecisionTraceState)


def _closed_reason(value: str) -> str | None:
    if value in _CLOSED_REASONS:
        return value
    return None


def _closed_state(value: str) -> str | None:
    if value in _CLOSED_STATES:
        return value
    return None


__all__ = [
    "AttributionEvidence",
    "AttributionEvidenceQuery",
    "AttributionEvidenceRecord",
]
