"""Allowlisted attribution evidence over the current false-positive cohort."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from shared.edge_db.event_neighborhood import (
    EXPECTED_NEIGHBORHOOD_FRAMES,
    NeighborhoodCoverage,
    NeighborhoodTrigger,
    coverage_for_decision,
)
from worker.fp_attribution.cohort import (
    FalsePositiveCohort,
    FalsePositiveCohortExclusion,
    FalsePositiveCohortMember,
    FalsePositiveCohortQuery,
    open_query_only_connection,
)
from worker.types.trace import DecisionTraceReason, DecisionTraceState

_SCORE_NAME = "fall_probability"
_THRESHOLD_NAME = "operating_threshold"
_VALUE_ABSENT = "value_not_persisted"
_FACT_NOT_PERSISTED = "value_not_persisted"
_FACT_UNPARSEABLE = "value_unparseable"
_FACT_MISALIGNED = "identity_or_domain_misaligned"
_FACT_NOT_APPLICABLE = "not_applicable"
_FACT_AMBIGUOUS = "identity_or_domain_ambiguous"
_FACT_STATE_MISSING = "observation_state_missing"
_FACT_STATE_NOT_APPLICABLE = "observation_state_not_applicable"
_FALL_DOMAIN = "fall"
_BED_DOMAIN = "bed_exit"
_POSE_COMPONENT_PREFIXES = ("pose.", "person.")
_CLOSED_OBSERVATION_STATES = frozenset(
    {"observed", "executed", "not-applicable", "not-scheduled", "missing"}
)
_CLOSED_DOMAINS = frozenset({_FALL_DOMAIN, _BED_DOMAIN})
_FALL_MODULES = frozenset({"fall.v1"})
_BED_MODULES = frozenset({"bed_exit.v1"})
EvidenceStatus = Literal["COMPLETE", "PRUNED", "UNKNOWN"]
PersonPresenceStatus = Literal["PERSON_FOUND", "PERSON_GAP"]
DueSignalStatus = Literal["DUE", "NOT_DUE"]
DomainFactStatus = Literal["AVAILABLE", "NOT_APPLICABLE"]
AlignmentStatus = Literal["ALIGNED", "MISALIGNED"]


@dataclass(frozen=True, slots=True)
class PersonPresenceEvidence:
    status: PersonPresenceStatus | None
    duration_frames: int | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class DueSignalEvidence:
    status: DueSignalStatus | None
    not_scheduled_frames: int | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class FallLatchEvidence:
    status: DomainFactStatus | None
    same_track: bool | None
    same_domain: bool | None
    rise_before_rearm: bool | None
    rearm_frames: int | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class BedStateEvidence:
    status: DomainFactStatus | None
    sequence: tuple[str, ...] | None
    durations_frames: tuple[int, ...] | None
    same_track: bool | None
    same_domain: bool | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class TrackStalenessEvidence:
    last_seen_offset_frames: int | None
    same_track: bool | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class DomainAlignmentEvidence:
    status: AlignmentStatus | None
    domain: str | None
    same_track: bool | None
    same_domain: bool | None
    same_camera_boot_epoch: bool | None
    missing_reason: str | None


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
    person_presence: PersonPresenceEvidence = PersonPresenceEvidence(
        None, None, _FACT_NOT_PERSISTED
    )
    due_signal: DueSignalEvidence = DueSignalEvidence(None, None, _FACT_NOT_PERSISTED)
    fall_latch: FallLatchEvidence = FallLatchEvidence(
        None, None, None, None, None, _FACT_NOT_PERSISTED
    )
    bed_state: BedStateEvidence = BedStateEvidence(
        None, None, None, None, None, _FACT_NOT_PERSISTED
    )
    track_staleness: TrackStalenessEvidence = TrackStalenessEvidence(
        None, None, _FACT_NOT_PERSISTED
    )
    domain_alignment: DomainAlignmentEvidence = DomainAlignmentEvidence(
        None, None, None, None, None, _FACT_NOT_PERSISTED
    )
    boot_changed: bool | None = None
    boot_changed_missing_reason: str | None = _FACT_NOT_PERSISTED
    epoch_changed: bool | None = None
    epoch_changed_missing_reason: str | None = _FACT_NOT_PERSISTED

    def machine_fields(self) -> dict[str, object]:
        return {
            "bed_state": asdict(self.bed_state),
            "boot_changed": self.boot_changed,
            "boot_changed_missing_reason": self.boot_changed_missing_reason,
            "domain_alignment": asdict(self.domain_alignment),
            "due_signal": asdict(self.due_signal),
            "epoch_changed": self.epoch_changed,
            "epoch_changed_missing_reason": self.epoch_changed_missing_reason,
            "fall_latch": asdict(self.fall_latch),
            "person_presence": asdict(self.person_presence),
            "track_staleness": asdict(self.track_staleness),
        }


@dataclass(frozen=True, slots=True)
class AttributionEvidence:
    records: tuple[AttributionEvidenceRecord, ...]
    exclusions: tuple[FalsePositiveCohortExclusion, ...]


class AttributionEvidenceQuery:
    """Compose one allowlisted evidence record per current cohort member."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def extract(
        self,
        connection: sqlite3.Connection | None = None,
        *,
        cohort: FalsePositiveCohort | None = None,
    ) -> AttributionEvidence:
        if connection is None:
            loaded = (
                cohort
                if cohort is not None
                else FalsePositiveCohortQuery(self.database_path).load()
            )
            owned = open_query_only_connection(self.database_path)
            try:
                records = tuple(_record_for(owned, member) for member in loaded.members)
            finally:
                owned.close()
            return AttributionEvidence(records=records, exclusions=loaded.exclusions)
        loaded = (
            cohort
            if cohort is not None
            else FalsePositiveCohortQuery(self.database_path).load(connection)
        )
        records = tuple(_record_for(connection, member) for member in loaded.members)
        return AttributionEvidence(records=records, exclusions=loaded.exclusions)


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
    domain = None if decision is None else _closed_domain(decision[7])
    decision_tokens_trusted = decision is None or (
        reason is not None and previous_state is not None and current_state is not None
    )
    domain_evidence = _domain_evidence(
        connection,
        trigger,
        track_id=track_id,
        domain=domain,
        current_state=current_state,
        coverage_complete=not coverage.neighborhood_pruned,
    )
    boot_changed, boot_reason, epoch_changed, epoch_reason = _identity_change(
        trigger,
        coverage_complete=not coverage.neighborhood_pruned,
    )
    if coverage.neighborhood_pruned:
        status: EvidenceStatus = "PRUNED"
        eligible = False
    elif not decision_tokens_trusted or _score_threshold_required(domain, score, threshold):
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
        boot_changed=boot_changed,
        boot_changed_missing_reason=boot_reason,
        epoch_changed=epoch_changed,
        epoch_changed_missing_reason=epoch_reason,
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
        person_presence=domain_evidence.person_presence,
        due_signal=domain_evidence.due_signal,
        fall_latch=domain_evidence.fall_latch,
        bed_state=domain_evidence.bed_state,
        track_staleness=domain_evidence.track_staleness,
        domain_alignment=domain_evidence.domain_alignment,
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
) -> tuple[str, str, str, int | None, str | None, int | None, str | None, str] | None:
    if decision_trace_id is None:
        return None
    row = connection.execute(
        """
        SELECT reason, previous_state, current_state, track_id, track_missing_reason,
               bed_id, bed_missing_reason, module_qualified_id
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
        _required_text(row[7]),
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


@dataclass(frozen=True, slots=True)
class _DomainEvidence:
    person_presence: PersonPresenceEvidence
    due_signal: DueSignalEvidence
    fall_latch: FallLatchEvidence
    bed_state: BedStateEvidence
    track_staleness: TrackStalenessEvidence
    domain_alignment: DomainAlignmentEvidence


@dataclass(frozen=True, slots=True)
class _NeighborhoodDecision:
    frame_seq: int
    reason: str | None
    previous_state: str | None
    current_state: str | None
    track_id: int | None
    domain: str | None
    same_camera_boot_epoch: bool
    parseable: bool


def _domain_evidence(
    connection: sqlite3.Connection,
    trigger: NeighborhoodTrigger | None,
    *,
    track_id: int | None,
    domain: str | None,
    current_state: str | None,
    coverage_complete: bool,
) -> _DomainEvidence:
    if trigger is None or not coverage_complete:
        return _unavailable_domain_evidence(_FACT_NOT_PERSISTED)
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
    analysis_rows = connection.execute(
        """
        SELECT analysis.frame_seq, persons.track_id
        FROM runtime_analysis_traces AS analysis
        LEFT JOIN runtime_analysis_persons AS persons
          ON persons.analysis_trace_id = analysis.trace_id
        WHERE analysis.worker_boot_id = ?
          AND analysis.camera_id = ?
          AND analysis.stream_epoch = ?
          AND analysis.frame_seq >= ?
          AND analysis.frame_seq <= ?
        ORDER BY analysis.frame_seq, persons.ordinal
        """,
        identity,
    ).fetchall()
    component_rows = connection.execute(
        """
        SELECT analysis.frame_seq, components.component_qualified_id,
               components.observation_state
        FROM runtime_analysis_traces AS analysis
        JOIN runtime_analysis_components AS components
          ON components.analysis_trace_id = analysis.trace_id
        WHERE analysis.worker_boot_id = ?
          AND analysis.camera_id = ?
          AND analysis.stream_epoch = ?
          AND analysis.frame_seq >= ?
          AND analysis.frame_seq <= ?
        ORDER BY analysis.frame_seq, components.ordinal
        """,
        identity,
    ).fetchall()
    decision_rows = connection.execute(
        """
        SELECT analysis.frame_seq, decision.reason, decision.previous_state,
               decision.current_state, decision.track_id, decision.module_qualified_id,
               analysis.worker_boot_id, analysis.camera_id, analysis.stream_epoch
        FROM runtime_analysis_traces AS analysis
        JOIN evidence_decision_traces AS decision
          ON decision.analysis_trace_id = analysis.trace_id
        WHERE analysis.worker_boot_id = ?
          AND analysis.camera_id = ?
          AND analysis.stream_epoch = ?
          AND analysis.frame_seq >= ?
          AND analysis.frame_seq <= ?
        ORDER BY analysis.frame_seq, decision.trace_id
        """,
        identity,
    ).fetchall()
    return _project_domain_evidence(
        trigger=trigger,
        track_id=track_id,
        domain=domain,
        current_state=current_state,
        analysis_rows=analysis_rows,
        component_rows=component_rows,
        decision_rows=decision_rows,
    )


def _project_domain_evidence(
    *,
    trigger: NeighborhoodTrigger,
    track_id: int | None,
    domain: str | None,
    current_state: str | None,
    analysis_rows: list[tuple[object, ...]],
    component_rows: list[tuple[object, ...]],
    decision_rows: list[tuple[object, ...]],
) -> _DomainEvidence:
    tracks_by_seq: dict[int, set[int | None]] = {}
    for row in analysis_rows:
        seq = _required_int(row[0])
        tracks_by_seq.setdefault(seq, set()).add(_optional_int(row[1]))
    person_presence = _person_presence(tracks_by_seq, trigger.frame_seq)
    due_signal = _due_signal(component_rows)
    decisions = tuple(_neighborhood_decision(row, trigger) for row in decision_rows)
    fall_latch = _fall_latch(decisions, track_id, domain, trigger.frame_seq)
    bed_state = _bed_state(decisions, track_id, domain, current_state, trigger.frame_seq)
    track_staleness = _track_staleness(tracks_by_seq, track_id, trigger.frame_seq)
    domain_alignment = _domain_alignment(
        decisions,
        track_id,
        domain,
        tracks_by_seq.get(trigger.frame_seq, set()),
    )
    return _DomainEvidence(
        person_presence=person_presence,
        due_signal=due_signal,
        fall_latch=fall_latch,
        bed_state=bed_state,
        track_staleness=track_staleness,
        domain_alignment=domain_alignment,
    )


def _unavailable_domain_evidence(reason: str) -> _DomainEvidence:
    return _DomainEvidence(
        person_presence=PersonPresenceEvidence(None, None, reason),
        due_signal=DueSignalEvidence(None, None, reason),
        fall_latch=FallLatchEvidence(None, None, None, None, None, reason),
        bed_state=BedStateEvidence(None, None, None, None, None, reason),
        track_staleness=TrackStalenessEvidence(None, None, reason),
        domain_alignment=DomainAlignmentEvidence(None, None, None, None, None, reason),
    )


def _person_presence(
    tracks_by_seq: dict[int, set[int | None]],
    trigger_seq: int,
) -> PersonPresenceEvidence:
    gap_frames = 0
    found_frames = 0
    for seq, tracks in tracks_by_seq.items():
        if seq >= trigger_seq:
            continue
        if not tracks or all(track is None for track in tracks):
            gap_frames += 1
        elif any(track is not None for track in tracks):
            found_frames += 1
    if gap_frames:
        return PersonPresenceEvidence("PERSON_GAP", gap_frames, None)
    if found_frames:
        return PersonPresenceEvidence("PERSON_FOUND", 0, None)
    return PersonPresenceEvidence(None, None, _FACT_NOT_PERSISTED)


def _due_signal(component_rows: list[tuple[object, ...]]) -> DueSignalEvidence:
    due_frames: set[int] = set()
    not_due_frames: set[int] = set()
    for row in component_rows:
        qualified = _required_text(row[1])
        if not _is_pose_component(qualified):
            continue
        state = _required_text(row[2])
        if state not in _CLOSED_OBSERVATION_STATES:
            return DueSignalEvidence(None, None, _FACT_UNPARSEABLE)
        if state == "missing":
            return DueSignalEvidence(None, None, _FACT_STATE_MISSING)
        if state == "not-applicable":
            return DueSignalEvidence(None, None, _FACT_STATE_NOT_APPLICABLE)
        seq = _required_int(row[0])
        if state == "not-scheduled":
            not_due_frames.add(seq)
        elif state in {"observed", "executed"}:
            due_frames.add(seq)
        else:
            return DueSignalEvidence(None, None, _FACT_UNPARSEABLE)
    if not due_frames and not not_due_frames:
        return DueSignalEvidence(None, None, _FACT_NOT_PERSISTED)
    if not_due_frames:
        return DueSignalEvidence("NOT_DUE", len(not_due_frames), None)
    return DueSignalEvidence("DUE", 0, None)


def _is_pose_component(qualified_id: str) -> bool:
    component_id = qualified_id.split(".sha256.", 1)[0]
    return component_id in {"pose", "person"} or qualified_id.startswith(_POSE_COMPONENT_PREFIXES)


def _neighborhood_decision(
    row: tuple[object, ...],
    trigger: NeighborhoodTrigger,
) -> _NeighborhoodDecision:
    reason = _closed_reason(_required_text(row[1]))
    previous_state = _closed_state(_required_text(row[2]))
    current_state = _closed_state(_required_text(row[3]))
    domain = _closed_domain(_required_text(row[5]))
    same_identity = (
        _required_text(row[6]) == trigger.worker_boot_id
        and _required_text(row[7]) == trigger.camera_id
        and _required_int(row[8]) == trigger.stream_epoch
    )
    parseable = reason is not None and previous_state is not None and current_state is not None
    return _NeighborhoodDecision(
        frame_seq=_required_int(row[0]),
        reason=reason,
        previous_state=previous_state,
        current_state=current_state,
        track_id=_optional_int(row[4]),
        domain=domain,
        same_camera_boot_epoch=same_identity,
        parseable=parseable,
    )


def _fall_latch(
    decisions: tuple[_NeighborhoodDecision, ...],
    track_id: int | None,
    domain: str | None,
    trigger_seq: int,
) -> FallLatchEvidence:
    if domain != _FALL_DOMAIN:
        return FallLatchEvidence("NOT_APPLICABLE", None, None, None, None, _FACT_NOT_APPLICABLE)
    aligned = tuple(
        item
        for item in decisions
        if item.same_camera_boot_epoch
        and item.domain == _FALL_DOMAIN
        and item.track_id is not None
        and item.track_id == track_id
        and item.parseable
    )
    if not aligned:
        return FallLatchEvidence(None, None, None, None, None, _FACT_NOT_PERSISTED)
    rise = next(
        (
            item
            for item in aligned
            if item.previous_state == DecisionTraceState.CLEAR.value
            and item.current_state == DecisionTraceState.FALL.value
            and item.frame_seq < trigger_seq
        ),
        None,
    )
    rearm = next(
        (
            item
            for item in aligned
            if item.previous_state == DecisionTraceState.FALL.value
            and item.current_state == DecisionTraceState.CLEAR.value
            and rise is not None
            and item.frame_seq > rise.frame_seq
            and item.frame_seq < trigger_seq
        ),
        None,
    )
    if rise is None or rearm is None:
        return FallLatchEvidence(None, None, None, None, None, _FACT_NOT_PERSISTED)
    return FallLatchEvidence(
        "AVAILABLE",
        True,
        True,
        True,
        trigger_seq - rearm.frame_seq,
        None,
    )


def _bed_state(
    decisions: tuple[_NeighborhoodDecision, ...],
    track_id: int | None,
    domain: str | None,
    current_state: str | None,
    trigger_seq: int,
) -> BedStateEvidence:
    if domain != _BED_DOMAIN:
        return BedStateEvidence("NOT_APPLICABLE", None, None, None, None, _FACT_NOT_APPLICABLE)
    aligned = tuple(
        item
        for item in decisions
        if item.same_camera_boot_epoch
        and item.domain == _BED_DOMAIN
        and item.track_id is not None
        and item.track_id == track_id
        and item.parseable
        and item.current_state is not None
    )
    if not aligned:
        return BedStateEvidence(None, None, None, None, None, _FACT_NOT_PERSISTED)
    sequence: list[str] = []
    starts: list[int] = []
    for item in aligned:
        state = item.current_state
        if state is None:
            continue
        if not sequence or sequence[-1] != state:
            sequence.append(state)
            starts.append(item.frame_seq)
    if current_state is not None and (not sequence or sequence[-1] != current_state):
        sequence.append(current_state)
        starts.append(trigger_seq)
    if len(sequence) < 2:
        return BedStateEvidence(None, None, None, None, None, _FACT_NOT_PERSISTED)
    ends = [*starts[1:], trigger_seq + 1]
    durations = tuple(end - start for start, end in zip(starts, ends, strict=True))
    return BedStateEvidence("AVAILABLE", tuple(sequence), durations, True, True, None)


def _track_staleness(
    tracks_by_seq: dict[int, set[int | None]],
    track_id: int | None,
    trigger_seq: int,
) -> TrackStalenessEvidence:
    if track_id is None:
        return TrackStalenessEvidence(None, None, _FACT_NOT_PERSISTED)
    last_seen: int | None = None
    for seq in sorted(seq for seq in tracks_by_seq if seq < trigger_seq):
        if track_id in tracks_by_seq[seq]:
            last_seen = seq
    if last_seen is None:
        return TrackStalenessEvidence(None, None, _FACT_NOT_PERSISTED)
    return TrackStalenessEvidence(trigger_seq - last_seen - 1, True, None)


def _domain_alignment(
    decisions: tuple[_NeighborhoodDecision, ...],
    track_id: int | None,
    domain: str | None,
    trigger_tracks: set[int | None],
) -> DomainAlignmentEvidence:
    if domain is None:
        return DomainAlignmentEvidence(None, None, None, None, None, _FACT_NOT_PERSISTED)
    extra_tracks = {track for track in trigger_tracks if track is not None and track != track_id}
    ambiguous_decisions = any(
        item.track_id != track_id or item.domain != domain or not item.same_camera_boot_epoch
        for item in decisions
    )
    if extra_tracks or ambiguous_decisions:
        return DomainAlignmentEvidence(None, domain, None, None, None, _FACT_AMBIGUOUS)
    return DomainAlignmentEvidence("ALIGNED", domain, True, True, True, None)


def _identity_change(
    trigger: NeighborhoodTrigger | None,
    *,
    coverage_complete: bool,
) -> tuple[bool | None, str | None, bool | None, str | None]:
    if trigger is None or not coverage_complete:
        return None, _FACT_NOT_PERSISTED, None, _FACT_NOT_PERSISTED
    return False, None, False, None


def _score_threshold_required(
    domain: str | None,
    score: float | None,
    threshold: float | None,
) -> bool:
    if domain == _BED_DOMAIN:
        return False
    return score is None or threshold is None


def _closed_domain(module_qualified_id: str) -> str | None:
    if module_qualified_id in _FALL_MODULES:
        return _FALL_DOMAIN
    if module_qualified_id in _BED_MODULES:
        return _BED_DOMAIN
    return None


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
    "BedStateEvidence",
    "DomainAlignmentEvidence",
    "DueSignalEvidence",
    "FallLatchEvidence",
    "PersonPresenceEvidence",
    "TrackStalenessEvidence",
]
