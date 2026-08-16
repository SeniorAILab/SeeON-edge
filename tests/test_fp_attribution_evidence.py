from __future__ import annotations

import ast
import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from shared.edge_db.migrator import migrate_database
from worker.fp_attribution import FalsePositiveCohortQuery

NOW = "2026-08-13T12:00:00Z"
LATER = "2026-08-13T12:01:00Z"
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
CAMERA_ID = "camera:alpha"
OTHER_CAMERA_ID = "camera:beta"
BOOT_ID = "boot:one"
OTHER_BOOT_ID = "boot:two"
EPOCH = 3
OTHER_EPOCH = 4
TRIGGER_SEQ = 40
PRECEDING_FRAMES = 29
NEIGHBORHOOD_SIZE = 30
NOTE_SENTINEL = "NOTE_SENTINEL_fp_evidence_7c21"
ACTOR_SENTINEL = "actor:sentinel-fp-evidence"
PAYLOAD_SENTINEL = "PAYLOAD_SENTINEL_fp_json_9e44"
PATH_SENTINEL = "/tmp/seeon-forbidden/clip.mp4"
GEOMETRY_SENTINEL = "polygon:[[1.25,9.5],[3.5,8.25]]"
COORD_SENTINEL = 987654

_FORBIDDEN_SOURCE_TOKENS = (
    "notes",
    "actor_id",
    "payload_json",
    "contained_relpath",
    "media_relpath",
    "manifest_path",
    "polygon",
    "geometry",
    "x1",
    "y1",
    "backend.app",
)
_ALLOWED_EVIDENCE_FIELDS = frozenset(
    {
        "edge_event_id",
        "decision_reason",
        "previous_state",
        "current_state",
        "score",
        "threshold",
        "score_missing_reason",
        "threshold_missing_reason",
        "track_id",
        "track_missing_reason",
        "track_changed",
        "bed_id",
        "bed_missing_reason",
        "bed_changed",
        "worker_boot_id",
        "stream_epoch",
        "boot_changed",
        "boot_changed_missing_reason",
        "epoch_changed",
        "epoch_changed_missing_reason",
        "associated_sibling_event_ids",
        "attempt_count",
        "backend_event_ids",
        "coverage_status",
        "coverage_reason",
        "expected_frames",
        "retained_frames",
        "neighborhood_pruned",
        "evidence_status",
        "category",
        "prevented_eligible",
        "person_presence",
        "due_signal",
        "fall_latch",
        "bed_state",
        "track_staleness",
        "domain_alignment",
    }
)


def _trace_id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _migrated(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _seed_manifest(
    connection: sqlite3.Connection,
    *,
    boot_id: str = BOOT_ID,
    camera_id: str = CAMERA_ID,
    applied_at: str = NOW,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_contents VALUES (?, 1, '{}', ?)",
        (MANIFEST_ID, applied_at),
    )
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_boots VALUES (?, ?, ?)",
        (boot_id, MANIFEST_ID, applied_at),
    )
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_cameras VALUES (?, ?, ?, ?)",
        (boot_id, camera_id, MANIFEST_ID, applied_at),
    )


def _insert_analysis(
    connection: sqlite3.Connection,
    *,
    seq: int,
    boot_id: str = BOOT_ID,
    camera_id: str = CAMERA_ID,
    epoch: int = EPOCH,
    track_id: int | None = 7,
    components: tuple[tuple[str, str], ...] = (),
) -> str:
    trace_id = _trace_id(f"analysis:{boot_id}:{camera_id}:{epoch}:{seq}")
    connection.execute(
        """
        INSERT INTO runtime_analysis_traces (
            trace_id, trace_schema_version, worker_boot_id, camera_id,
            stream_epoch, frame_seq, pts, source_time_sec, frame_width,
            frame_height, bed_region_provenance, storage_bytes
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, 16, 16, 'fresh', 1)
        """,
        (trace_id, boot_id, camera_id, epoch, seq, float(seq), float(seq)),
    )
    if track_id is None:
        connection.execute(
            """
            INSERT INTO runtime_analysis_persons (
                analysis_trace_id, ordinal, track_id, track_missing_reason,
                x1, y1, x2, y2, confidence
            ) VALUES (?, 0, NULL, 'no-observed-person', ?, ?, ?, ?, 0.1)
            """,
            (
                trace_id,
                COORD_SENTINEL,
                COORD_SENTINEL,
                COORD_SENTINEL + 1,
                COORD_SENTINEL + 1,
            ),
        )
    else:
        connection.execute(
            """
            INSERT INTO runtime_analysis_persons (
                analysis_trace_id, ordinal, track_id, track_missing_reason,
                x1, y1, x2, y2, confidence
            ) VALUES (?, 0, ?, NULL, ?, ?, ?, ?, 0.9)
            """,
            (
                trace_id,
                track_id,
                COORD_SENTINEL,
                COORD_SENTINEL,
                COORD_SENTINEL + 1,
                COORD_SENTINEL + 1,
            ),
        )
    for ordinal, (qualified_id, observation_state) in enumerate(components):
        connection.execute(
            "INSERT INTO runtime_analysis_components "
            "(analysis_trace_id, ordinal, component_qualified_id, observation_state) "
            "VALUES (?, ?, ?, ?)",
            (trace_id, ordinal, qualified_id, observation_state),
        )
    return trace_id


def _insert_decision(
    connection: sqlite3.Connection,
    *,
    decision_trace_id: str,
    analysis_trace_id: str | None,
    reason: str = "fall-onset",
    previous_state: str = "clear",
    current_state: str = "fall",
    track_id: int | None = 7,
    track_missing_reason: str | None = None,
    bed_id: int | None = None,
    bed_missing_reason: str | None = "not-applicable",
    module_qualified_id: str = "fall.v1",
    policy_qualified_id: str = "fall.policy.v1",
    triggered: int = 1,
    values: tuple[tuple[str, float | None, str | None], ...] = (
        ("fall_probability", 0.91, None),
        ("operating_threshold", 0.5, None),
    ),
) -> str:
    connection.execute(
        """
        INSERT INTO evidence_decision_traces (
            trace_id, trace_schema_version, analysis_trace_id,
            module_qualified_id, policy_qualified_id, effective_policy_id,
            runtime_manifest_sha256, reason, previous_state, current_state,
            triggered, track_id, track_missing_reason, bed_id, bed_missing_reason
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_trace_id,
            analysis_trace_id,
            module_qualified_id,
            policy_qualified_id,
            POLICY_ID,
            MANIFEST_ID,
            reason,
            previous_state,
            current_state,
            triggered,
            track_id,
            track_missing_reason,
            bed_id,
            bed_missing_reason,
        ),
    )
    if values:
        connection.executemany(
            "INSERT INTO evidence_decision_values "
            "(decision_trace_id, name, numeric_value, missing_reason) VALUES (?, ?, ?, ?)",
            tuple((decision_trace_id, name, numeric, missing) for name, numeric, missing in values),
        )
    return decision_trace_id


def _insert_event(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    attempt_count: int = 1,
    backend_event_id: str | None = "backend:alpha",
) -> None:
    payload = (
        f'{{"secret":"{PAYLOAD_SENTINEL}","path":"{PATH_SENTINEL}",'
        f'"geometry":"{GEOMETRY_SENTINEL}"}}'
    )
    connection.execute(
        """
        INSERT INTO evidence_events (
            edge_event_id, detected_at, payload_json, state, queued_at,
            next_attempt_at, attempt_count, delivery_state, backend_event_id
        ) VALUES (?, ?, ?, 'ACKED', 1, 1, ?, 'ACKED', ?)
        """,
        (edge_event_id, NOW, payload, attempt_count, backend_event_id),
    )


def _insert_clip(connection: sqlite3.Connection, clip_id: str) -> None:
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, state_version) "
        "VALUES (?, 'VERIFIED', 1)",
        (clip_id,),
    )


def _insert_incident(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    edge_event_id: str,
    clip_id: str | None,
    decision_trace_id: str | None,
    camera_id: str = CAMERA_ID,
    event_type: str = "fall",
    module_qualified_id: str = "fall.v1",
    policy_qualified_id: str = "fall.policy.v1",
) -> None:
    if decision_trace_id is None:
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, primary_clip_id, lifecycle_state,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'NOT_RECORDED', ?, 'STAGING', ?, ?)
            """,
            (incident_id, edge_event_id, camera_id, event_type, NOW, clip_id, NOW, NOW),
        )
        return
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            runtime_manifest_sha256, decision_trace_id, module_qualified_id,
            policy_qualified_id, effective_policy_id, provenance_state,
            primary_clip_id, lifecycle_state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  'QUALIFIED', ?, 'STAGING', ?, ?)
        """,
        (
            incident_id,
            edge_event_id,
            camera_id,
            event_type,
            NOW,
            MANIFEST_ID,
            decision_trace_id,
            module_qualified_id,
            policy_qualified_id,
            POLICY_ID,
            clip_id,
            NOW,
            NOW,
        ),
    )


def _insert_primary(connection: sqlite3.Connection, *, incident_id: str, clip_id: str) -> None:
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, 'NOT_RECORDED', '[]', 'MISSING', ?)
        """,
        (incident_id, clip_id, NOW),
    )


def _insert_review(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    clip_id: str,
    disposition: str = "FALSE_POSITIVE",
    version: int = 1,
) -> None:
    connection.execute(
        """
        INSERT INTO control_evidence_review_revisions (
            review_id, incident_id, clip_id, review_version, actor_id,
            reviewed_at, disposition, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"review:{incident_id}:{version}",
            incident_id,
            clip_id,
            version,
            ACTOR_SENTINEL,
            NOW if version == 1 else LATER,
            disposition,
            NOTE_SENTINEL,
        ),
    )
    existing = connection.execute(
        "SELECT current_version FROM control_evidence_review_state WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO control_evidence_review_state "
            "(incident_id, clip_id, current_version) VALUES (?, ?, ?)",
            (incident_id, clip_id, version),
        )
        return
    connection.execute(
        "UPDATE control_evidence_review_state SET current_version = ? WHERE incident_id = ?",
        (version, incident_id),
    )


def _insert_cursor(
    connection: sqlite3.Connection,
    *,
    camera_id: str = CAMERA_ID,
    pruned_frames: int = 0,
    oldest_retained_seq: int | None = None,
    newest_retained_seq: int | None = None,
) -> None:
    oldest_trace = None if oldest_retained_seq is None else _trace_id("cursor-oldest")
    newest_trace = None if newest_retained_seq is None else _trace_id("cursor-newest")
    connection.execute(
        """
        INSERT INTO runtime_trace_cursors (
            camera_id, handoff_dropped_frames, pruned_frames,
            oldest_retained_seq, newest_retained_seq, updated_at_source_sec,
            persistence_failed_frames, retention_blocked_frames,
            oldest_retained_boot_id, oldest_retained_stream_epoch,
            oldest_retained_trace_id, newest_retained_boot_id,
            newest_retained_stream_epoch, newest_retained_trace_id
        ) VALUES (?, 0, ?, ?, ?, 40.0, 0, 0, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            pruned_frames,
            oldest_retained_seq,
            newest_retained_seq,
            None if oldest_retained_seq is None else BOOT_ID,
            None if oldest_retained_seq is None else EPOCH,
            oldest_trace,
            None if newest_retained_seq is None else BOOT_ID,
            None if newest_retained_seq is None else EPOCH,
            newest_trace,
        ),
    )


def _bind_clip_event(
    connection: sqlite3.Connection,
    *,
    clip_id: str,
    edge_event_id: str,
    ordinal: int,
) -> None:
    connection.execute(
        "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
        (clip_id, edge_event_id, ordinal),
    )


def _seed_fp_event(
    connection: sqlite3.Connection,
    *,
    suffix: str,
    seqs: tuple[int, ...],
    trigger_seq: int = TRIGGER_SEQ,
    extra_frames: tuple[tuple[str, str, int, int], ...] = (),
    extra_persons: tuple[tuple[str, str, int, int, int, int], ...] = (),
    cursor: dict[str, object] | None = None,
    include_decision: bool = True,
    values: tuple[tuple[str, float | None, str | None], ...] | None = (
        ("fall_probability", 0.91, None),
        ("operating_threshold", 0.5, None),
    ),
    track_id: int | None = 7,
    bed_id: int | None = None,
    attempt_count: int = 1,
    backend_event_id: str | None = "backend:alpha",
    clip_id: str | None = None,
    bind_clip: bool = False,
    clip_ordinal: int = 0,
    decision_trace_id: str | None = None,
    ref_trace_id: str | None = None,
    analysis_track_by_seq: dict[int, int | None] | None = None,
    disposition: str = "FALSE_POSITIVE",
    camera_id: str = CAMERA_ID,
    reason: str = "fall-onset",
    previous_state: str = "clear",
    current_state: str = "fall",
    module_qualified_id: str = "fall.v1",
    policy_qualified_id: str = "fall.policy.v1",
    event_type: str = "fall",
    triggered: int = 1,
    components_by_seq: dict[int, tuple[tuple[str, str], ...]] | None = None,
    neighborhood_decisions: tuple[dict[str, object], ...] = (),
) -> str:
    edge_event_id = f"event:{suffix}"
    incident_id = f"incident:{suffix}"
    review_clip_id = clip_id or f"clip:{suffix}"
    _seed_manifest(connection)
    _seed_manifest(connection, boot_id=OTHER_BOOT_ID, applied_at="2026-08-13T01:00:00Z")
    _seed_manifest(connection, camera_id=OTHER_CAMERA_ID)
    _seed_manifest(connection, camera_id=camera_id)
    trigger_analysis_id: str | None = None
    for seq in seqs:
        person_track = track_id if analysis_track_by_seq is None else analysis_track_by_seq.get(
            seq, track_id
        )
        analysis_id = _insert_analysis(
            connection,
            seq=seq,
            camera_id=camera_id,
            track_id=person_track,
            components=() if components_by_seq is None else components_by_seq.get(seq, ()),
        )
        if seq == trigger_seq:
            trigger_analysis_id = analysis_id
    for boot_id, extra_camera_id, epoch, seq in extra_frames:
        _insert_analysis(
            connection,
            seq=seq,
            boot_id=boot_id,
            camera_id=extra_camera_id,
            epoch=epoch,
        )
    for boot_id, extra_camera_id, epoch, seq, ordinal, extra_track in extra_persons:
        connection.execute(
            """
            INSERT INTO runtime_analysis_persons (
                analysis_trace_id, ordinal, track_id, track_missing_reason,
                x1, y1, x2, y2, confidence
            ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, 0.9)
            """,
            (
                _trace_id(f"analysis:{boot_id}:{extra_camera_id}:{epoch}:{seq}"),
                ordinal,
                extra_track,
                COORD_SENTINEL,
                COORD_SENTINEL,
                COORD_SENTINEL + 1,
                COORD_SENTINEL + 1,
            ),
        )
    resolved_decision = decision_trace_id or _trace_id(f"decision:{suffix}")
    if include_decision:
        _insert_decision(
            connection,
            decision_trace_id=resolved_decision,
            analysis_trace_id=trigger_analysis_id,
            reason=reason,
            previous_state=previous_state,
            current_state=current_state,
            track_id=track_id,
            track_missing_reason=None if track_id is not None else "not-applicable",
            bed_id=bed_id,
            bed_missing_reason=None if bed_id is not None else "not-applicable",
            module_qualified_id=module_qualified_id,
            policy_qualified_id=policy_qualified_id,
            triggered=triggered,
            values=() if values is None else values,
        )
        for extra in neighborhood_decisions:
            extra_seq = int(extra["seq"])
            extra_boot = str(extra.get("boot_id", BOOT_ID))
            extra_camera = str(extra.get("camera_id", camera_id))
            extra_epoch = int(extra.get("epoch", EPOCH))
            extra_track = extra.get("track_id", track_id)
            extra_bed = extra.get("bed_id", bed_id)
            extra_module = str(extra.get("module_qualified_id", module_qualified_id))
            extra_policy = str(extra.get("policy_qualified_id", policy_qualified_id))
            _insert_decision(
                connection,
                decision_trace_id=_trace_id(
                    f"decision:{suffix}:extra:{extra_boot}:{extra_camera}:{extra_epoch}:{extra_seq}"
                ),
                analysis_trace_id=_trace_id(
                    f"analysis:{extra_boot}:{extra_camera}:{extra_epoch}:{extra_seq}"
                ),
                reason=str(extra["reason"]),
                previous_state=str(extra["previous_state"]),
                current_state=str(extra["current_state"]),
                track_id=None if extra_track is None else int(extra_track),
                track_missing_reason=None if extra_track is not None else "not-applicable",
                bed_id=None if extra_bed is None else int(extra_bed),
                bed_missing_reason=None if extra_bed is not None else "not-applicable",
                module_qualified_id=extra_module,
                policy_qualified_id=extra_policy,
                triggered=int(extra.get("triggered", 0)),
                values=tuple(extra.get("values", ())),
            )
    _insert_event(
        connection,
        edge_event_id=edge_event_id,
        attempt_count=attempt_count,
        backend_event_id=backend_event_id,
    )
    _insert_clip(connection, review_clip_id)
    _insert_incident(
        connection,
        incident_id=incident_id,
        edge_event_id=edge_event_id,
        clip_id=review_clip_id,
        decision_trace_id=resolved_decision if include_decision else None,
        camera_id=camera_id,
        event_type=event_type,
        module_qualified_id=module_qualified_id,
        policy_qualified_id=policy_qualified_id,
    )
    _insert_primary(connection, incident_id=incident_id, clip_id=review_clip_id)
    if include_decision:
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
            (edge_event_id, ref_trace_id or resolved_decision),
        )
    elif ref_trace_id is not None:
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
            (edge_event_id, ref_trace_id),
        )
    _insert_review(
        connection,
        incident_id=incident_id,
        clip_id=review_clip_id,
        disposition=disposition,
    )
    if bind_clip:
        _bind_clip_event(
            connection,
            clip_id=review_clip_id,
            edge_event_id=edge_event_id,
            ordinal=clip_ordinal,
        )
    if cursor is not None:
        _insert_cursor(connection, **cursor)
    return edge_event_id


def _complete_seqs() -> tuple[int, ...]:
    return tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1))


def _extract(database: Path):
    from worker.fp_attribution import AttributionEvidenceQuery

    return AttributionEvidenceQuery(database).extract()


def _record_for(result: object, edge_event_id: str):
    matches = [item for item in result.records if item.edge_event_id == edge_event_id]
    assert len(matches) == 1
    return matches[0]


def _assert_ineligible(record: object) -> None:
    assert record.category is None
    assert record.prevented_eligible is False


def test_todo6_complete_and_pruned_coverage_still_holds(tmp_path: Path) -> None:
    """Characterize Todo 6 complete vs 29-row pruned behavior first.

    Given exact-30 and 29-row same-identity windows on a migrated database
    When the Todo 6 neighborhood query runs
    Then complete stays unpruned and the short window stays ineligible.
    """

    from backend.app.features.evidence.explanation_neighborhood import (
        EXPECTED_NEIGHBORHOOD_FRAMES,
        EventNeighborhoodQuery,
    )

    complete = _migrated(tmp_path / "complete")
    pruned = _migrated(tmp_path / "pruned")
    complete_decision = _trace_id("todo6-complete")
    pruned_decision = _trace_id("todo6-pruned")
    with _connect(complete) as connection:
        _seed_fp_event(
            connection,
            suffix="todo6-complete",
            seqs=_complete_seqs(),
            decision_trace_id=complete_decision,
        )
        connection.commit()
    with _connect(pruned) as connection:
        _seed_fp_event(
            connection,
            suffix="todo6-pruned",
            seqs=tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1)),
            decision_trace_id=pruned_decision,
        )
        connection.commit()

    complete_coverage = EventNeighborhoodQuery(complete).coverage_for_decision(complete_decision)
    pruned_coverage = EventNeighborhoodQuery(pruned).coverage_for_decision(pruned_decision)

    assert complete_coverage.status == "COMPLETE"
    assert complete_coverage.neighborhood_pruned is False
    assert complete_coverage.expected_frames == EXPECTED_NEIGHBORHOOD_FRAMES
    assert complete_coverage.retained_frames == NEIGHBORHOOD_SIZE
    assert complete_coverage.category is None
    assert pruned_coverage.status == "GAP"
    assert pruned_coverage.neighborhood_pruned is True
    assert pruned_coverage.retained_frames == 29
    assert pruned_coverage.prevented_eligible is False
    assert pruned_coverage.category is None


def test_todo10_current_fp_is_the_only_cohort_member(tmp_path: Path) -> None:
    """Characterize Todo 10: only the current FP revision is in the cohort.

    Given a current FP and a current TP on a migrated database
    When the cohort query loads
    Then only the current FP event is a member.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        current_fp = _seed_fp_event(
            connection,
            suffix="now-fp",
            seqs=_complete_seqs(),
        )
        current_tp = _seed_fp_event(
            connection,
            suffix="now-tp",
            seqs=tuple(range(70, 100)),
            trigger_seq=99,
            disposition="TRUE_POSITIVE",
        )
        connection.commit()

    cohort = FalsePositiveCohortQuery(database).load()

    assert tuple(member.edge_event_id for member in cohort.members) == (current_fp,)
    assert current_tp not in {member.edge_event_id for member in cohort.members}


def test_exact_30_complete_member_is_an_allowlisted_attributable_record(
    tmp_path: Path,
) -> None:
    """Exact-30 coverage plus decision facts yields an attributable-ready record.

    Given a current FP with a complete same-identity window and score/threshold
    When attribution evidence is extracted
    Then the record is COMPLETE, keep category null, and expose only allowlisted facts.
    """

    database = _migrated(tmp_path)
    extra = (
        tuple((OTHER_BOOT_ID, CAMERA_ID, EPOCH, seq) for seq in _complete_seqs())
        + tuple((BOOT_ID, OTHER_CAMERA_ID, EPOCH, seq) for seq in _complete_seqs())
        + tuple((BOOT_ID, CAMERA_ID, OTHER_EPOCH, seq) for seq in range(30))
    )
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="complete",
            seqs=_complete_seqs(),
            extra_frames=extra,
            attempt_count=3,
            backend_event_id="backend:one",
            track_id=7,
            analysis_track_by_seq={seq: 7 for seq in _complete_seqs()} | {25: 8},
        )
        connection.commit()

    result = _extract(database)
    record = _record_for(result, edge_event_id)

    assert record.edge_event_id == edge_event_id
    assert record.decision_reason == "fall-onset"
    assert record.previous_state == "clear"
    assert record.current_state == "fall"
    assert record.score == 0.91
    assert record.threshold == 0.5
    assert record.score_missing_reason is None
    assert record.threshold_missing_reason is None
    assert record.track_id == 7
    assert record.track_changed is True
    assert record.bed_id is None
    assert record.bed_missing_reason == "not-applicable"
    assert record.bed_changed is False
    assert record.worker_boot_id == BOOT_ID
    assert record.stream_epoch == EPOCH
    assert record.boot_changed is False
    assert record.epoch_changed is False
    assert record.associated_sibling_event_ids == ()
    assert record.attempt_count == 3
    assert record.backend_event_ids == ("backend:one",)
    assert record.coverage_status == "COMPLETE"
    assert record.coverage_reason is None
    assert record.expected_frames == NEIGHBORHOOD_SIZE
    assert record.retained_frames == NEIGHBORHOOD_SIZE
    assert record.neighborhood_pruned is False
    assert record.evidence_status == "COMPLETE"
    assert record.category is None
    assert record.prevented_eligible is True
    assert {item.name for item in fields(record)} == _ALLOWED_EVIDENCE_FIELDS


def test_twenty_nine_gap_prefix_retention_and_missing_trace_are_pruned(
    tmp_path: Path,
) -> None:
    """Incomplete windows and missing traces stay PRUNED, never UNKNOWN.

    Given 29-row, interior-gap, epoch-prefix, retention-loss, and no-trace FPs
    When attribution evidence is extracted
    Then every record is PRUNED with category null and cannot be classified.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        twenty_nine = _seed_fp_event(
            connection,
            suffix="twenty-nine",
            seqs=tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1)),
        )
        gap = _seed_fp_event(
            connection,
            suffix="gap",
            seqs=tuple(
                seq
                for seq in range(70, 100)
                if seq != 85
            ),
            trigger_seq=99,
        )
        prefix = _seed_fp_event(
            connection,
            suffix="prefix",
            seqs=tuple(range(11)),
            trigger_seq=10,
            extra_frames=tuple((BOOT_ID, CAMERA_ID, OTHER_EPOCH, seq) for seq in range(20, 50)),
        )
        retention = _seed_fp_event(
            connection,
            suffix="retention",
            seqs=tuple(range(120, 141)),
            trigger_seq=140,
            camera_id="camera:retention",
            cursor={
                "camera_id": "camera:retention",
                "pruned_frames": 9,
                "oldest_retained_seq": 120,
                "newest_retained_seq": 140,
            },
        )
        missing_trace = _seed_fp_event(
            connection,
            suffix="no-trace",
            seqs=(),
            include_decision=False,
        )
        connection.commit()

    result = _extract(database)
    by_id = {record.edge_event_id: record for record in result.records}

    expected = {
        twenty_nine: ("GAP", "NEIGHBORHOOD_GAP_UNEXPLAINED", 29),
        gap: ("GAP", "NEIGHBORHOOD_GAP_UNEXPLAINED", 29),
        prefix: ("EPOCH_PREFIX_SHORT", "NEIGHBORHOOD_EPOCH_PREFIX_SHORT", 11),
        retention: ("PRUNED", "NEIGHBORHOOD_PRUNED", 21),
        missing_trace: ("MISSING_TRIGGER", "DECISION_TRACE_NOT_RECORDED", 0),
    }
    assert set(by_id) == set(expected)
    for edge_event_id, (status, reason, retained) in expected.items():
        record = by_id[edge_event_id]
        assert record.neighborhood_pruned is True
        assert record.evidence_status == "PRUNED"
        assert record.coverage_status == status
        assert record.coverage_reason == reason
        assert record.expected_frames == NEIGHBORHOOD_SIZE
        assert record.retained_frames == retained
        _assert_ineligible(record)


def test_missing_decision_facts_on_complete_window_are_unknown(tmp_path: Path) -> None:
    """A complete window without score/threshold is UNKNOWN, not PRUNED.

    Given an exact-30 FP whose decision values were never persisted
    When attribution evidence is extracted
    Then coverage stays complete and the record is typed UNKNOWN.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="unknown-facts",
            seqs=_complete_seqs(),
            values=None,
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.neighborhood_pruned is False
    assert record.coverage_status == "COMPLETE"
    assert record.coverage_reason is None
    assert record.score is None
    assert record.threshold is None
    assert record.score_missing_reason == "value_not_persisted"
    assert record.threshold_missing_reason == "value_not_persisted"
    assert record.evidence_status == "UNKNOWN"
    _assert_ineligible(record)


def test_trace_conflict_stays_out_of_evaluable_evidence(tmp_path: Path) -> None:
    """Unequal incident/ref traces cannot become COMPLETE or UNKNOWN.

    Given a current FP whose incident and ref decision IDs disagree
    When attribution evidence is extracted
    Then the event stays a TRACE_REF_CONFLICT exclusion, not an evaluable record.
    """

    database = _migrated(tmp_path)
    incident_trace = _trace_id("conflict-a")
    ref_trace = _trace_id("conflict-b")
    with _connect(database) as connection:
        _seed_manifest(connection, camera_id=OTHER_CAMERA_ID)
        other_analysis = _insert_analysis(connection, seq=200, camera_id=OTHER_CAMERA_ID)
        _insert_decision(
            connection,
            decision_trace_id=ref_trace,
            analysis_trace_id=other_analysis,
        )
        _seed_fp_event(
            connection,
            suffix="conflict",
            seqs=_complete_seqs(),
            decision_trace_id=incident_trace,
            ref_trace_id=ref_trace,
        )
        connection.commit()

    result = _extract(database)

    assert result.records == ()
    assert "TRACE_REF_CONFLICT" in {item.reason for item in result.exclusions}


def test_cross_camera_boot_and_epoch_rows_cannot_complete_coverage(
    tmp_path: Path,
) -> None:
    """Foreign camera/boot/epoch rows never fill a missing predecessor.

    Given 29 same-identity rows and the missing seq on other identities
    When attribution evidence is extracted
    Then the member stays PRUNED with an unexplained gap.
    """

    database = _migrated(tmp_path)
    missing = TRIGGER_SEQ - PRECEDING_FRAMES
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="isolated",
            seqs=tuple(range(missing + 1, TRIGGER_SEQ + 1)),
            extra_frames=(
                (BOOT_ID, OTHER_CAMERA_ID, EPOCH, missing),
                (OTHER_BOOT_ID, CAMERA_ID, EPOCH, missing),
                (BOOT_ID, CAMERA_ID, OTHER_EPOCH, missing),
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.neighborhood_pruned is True
    assert record.evidence_status == "PRUNED"
    assert record.coverage_status == "GAP"
    assert record.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert record.retained_frames == 29
    assert record.worker_boot_id == BOOT_ID
    assert record.stream_epoch == EPOCH
    _assert_ineligible(record)


def test_only_explicit_clip_associated_siblings_are_emitted(tmp_path: Path) -> None:
    """Time-adjacent events are not siblings without a shared clip association.

    Given one complete FP sharing a clip with another event, plus a neighbor
    When attribution evidence is extracted
    Then only the clip-associated event ID is listed.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        complete = _seed_fp_event(
            connection,
            suffix="anchor",
            seqs=_complete_seqs(),
            clip_id="clip:shared",
            bind_clip=True,
            clip_ordinal=0,
        )
        sibling_decision = _trace_id("sibling-decision")
        sibling_analysis = _insert_analysis(connection, seq=80)
        _insert_decision(
            connection,
            decision_trace_id=sibling_decision,
            analysis_trace_id=sibling_analysis,
            bed_id=3,
            bed_missing_reason=None,
        )
        _insert_event(
            connection,
            edge_event_id="event:sibling",
            backend_event_id="backend:sibling",
        )
        _insert_incident(
            connection,
            incident_id="incident:sibling",
            edge_event_id="event:sibling",
            clip_id="clip:shared",
            decision_trace_id=sibling_decision,
        )
        _insert_primary(connection, incident_id="incident:sibling", clip_id="clip:shared")
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
            ("event:sibling", sibling_decision),
        )
        _bind_clip_event(
            connection,
            clip_id="clip:shared",
            edge_event_id="event:sibling",
            ordinal=1,
        )
        neighbor = _seed_fp_event(
            connection,
            suffix="neighbor",
            seqs=tuple(range(200, 230)),
            trigger_seq=229,
            clip_id="clip:other",
            bind_clip=True,
        )
        connection.commit()

    result = _extract(database)
    record = _record_for(result, complete)

    assert record.associated_sibling_event_ids == ("event:sibling",)
    assert neighbor not in record.associated_sibling_event_ids
    assert "event:neighbor" not in record.associated_sibling_event_ids
    assert _record_for(result, neighbor).associated_sibling_event_ids == ()


def test_multiple_attempts_remain_one_event(tmp_path: Path) -> None:
    """Attempt multiplicity is annotation, not a second evidence record.

    Given one current FP with three delivery attempts and one backend id
    When attribution evidence is extracted
    Then exactly one record exists for that edge_event_id.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="retries",
            seqs=_complete_seqs(),
            attempt_count=3,
            backend_event_id="backend:same",
        )
        connection.commit()

    result = _extract(database)
    matches = [record for record in result.records if record.edge_event_id == edge_event_id]

    assert len(matches) == 1
    assert matches[0].attempt_count == 3
    assert matches[0].backend_event_ids == ("backend:same",)
    assert matches[0].evidence_status == "COMPLETE"


def test_worker_reuses_shared_todo6_coverage_primitive_not_backend(
    tmp_path: Path,
) -> None:
    """Worker evidence must call the extracted Todo 6 primitive, not backend.

    Given the shared coverage module and the backend compatibility wrapper
    When worker evidence is imported
    Then both surfaces are the same object and worker source never imports backend.
    """

    from backend.app.features.evidence.explanation_neighborhood import (
        EventNeighborhoodQuery as BackendQuery,
    )
    from backend.app.features.evidence.explanation_neighborhood import (
        coverage_for_decision as backend_coverage,
    )
    from shared.edge_db.event_neighborhood import (
        EventNeighborhoodQuery as SharedQuery,
    )
    from shared.edge_db.event_neighborhood import (
        coverage_for_decision as shared_coverage,
    )
    from worker.fp_attribution import evidence as evidence_module

    source = Path(evidence_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert SharedQuery is BackendQuery
    assert shared_coverage is backend_coverage
    assert "shared.edge_db.event_neighborhood" in source
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(name == "backend" or name.startswith("backend.") for name in imported)
    assert "from backend" not in source
    assert "import backend" not in source


def test_evidence_rejects_malformed_database_path(tmp_path: Path) -> None:
    from worker.fp_attribution import AttributionEvidenceQuery

    query = AttributionEvidenceQuery(tmp_path / "missing.sqlite3")

    with pytest.raises(ValueError, match="edge-db"):
        query.extract()
    with pytest.raises(ValueError, match="edge-db"):
        AttributionEvidenceQuery(Path("")).extract()


def test_evidence_projection_never_exposes_notes_payload_path_or_geometry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from worker.fp_attribution import evidence as evidence_module

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        _seed_fp_event(connection, suffix="private", seqs=_complete_seqs())
        connection.commit()

    with caplog.at_level(logging.DEBUG):
        result = _extract(database)

    captured = capsys.readouterr()
    rendered = repr(asdict(result.records[0]))
    source = Path(evidence_module.__file__).read_text(encoding="utf-8")

    assert result.records[0].edge_event_id == "event:private"
    assert {item.name for item in fields(result.records[0])} == _ALLOWED_EVIDENCE_FIELDS
    for token in (
        NOTE_SENTINEL,
        ACTOR_SENTINEL,
        PAYLOAD_SENTINEL,
        PATH_SENTINEL,
        GEOMETRY_SENTINEL,
        str(COORD_SENTINEL),
    ):
        assert token not in rendered
        assert token not in caplog.text
        assert token not in captured.out
        assert token not in captured.err
        assert token not in source
    for token in _FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
    assert "SELECT" in source.upper()
    assert "payload_json" not in source
    assert "notes" not in source


def _selected_rtsp_sentinel() -> str:
    return "".join(
        (
            "rtsp",
            "://",
            "user",
            ":",
            "EVIDENCE_pass_9e44",
            "@",
            "10.255.255.3",
            "/stream",
        )
    )


def test_schema_valid_selected_decision_text_is_closed_or_unknown(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from worker.fp_attribution import classify_record
    from worker.fp_attribution import evidence as evidence_module

    # Given a current FP whose selected reason/state columns are schema-valid RTSP text
    secret = _selected_rtsp_sentinel()
    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(connection, suffix="poisoned-text", seqs=_complete_seqs())
        updated = connection.execute(
            "UPDATE evidence_decision_traces "
            "SET reason = ?, previous_state = ?, current_state = ?",
            (secret, secret, secret),
        ).rowcount
        assert updated == 1
        connection.commit()

    with caplog.at_level(logging.DEBUG):
        result = _extract(database)
    record = _record_for(result, edge_event_id)
    decision = classify_record(record)
    captured = capsys.readouterr()
    rendered = repr(asdict(record)) + repr(decision)
    source = Path(evidence_module.__file__).read_text(encoding="utf-8")

    # Then invalid selected text becomes typed unavailable and cannot influence attribution
    assert record.decision_reason is None
    assert record.previous_state is None
    assert record.current_state is None
    assert record.evidence_status == "UNKNOWN"
    assert record.neighborhood_pruned is False
    _assert_ineligible(record)
    assert decision.category is None
    assert decision.evidence_status == "UNKNOWN"
    assert secret not in rendered
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert secret not in source
    leaked = json.dumps({"decision_reason": secret, "previous_state": secret})
    with pytest.raises(AssertionError):
        assert secret not in leaked
        assert secret not in rendered


@pytest.mark.parametrize(
    "secret",
    (
        "/private/evidence-reason-path.bin",
        "ghp_" + ("D" * 36),
        "q" * 257,
        "clear\x07",
        "f\u0430ll",
    ),
    ids=(
        "absolute_path",
        "token_like",
        "overlength",
        "control_chars",
        "unicode_confusable",
    ),
)
def test_hostile_selected_decision_text_stays_unknown(
    tmp_path: Path,
    secret: str,
) -> None:
    from worker.fp_attribution import classify_record

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="hostile-text",
            seqs=_complete_seqs(),
        )
        updated = connection.execute(
            "UPDATE evidence_decision_traces "
            "SET reason = ?, previous_state = ?, current_state = ?",
            (secret, secret, secret),
        ).rowcount
        assert updated == 1
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)
    decision = classify_record(record)
    rendered = repr(asdict(record)) + repr(decision)

    assert record.decision_reason is None
    assert record.previous_state is None
    assert record.current_state is None
    assert record.evidence_status == "UNKNOWN"
    _assert_ineligible(record)
    assert decision.category is None
    assert secret not in rendered


_POSE_COMPONENT = "pose.sha256." + ("c" * 64)
_PERSON_COMPONENT = "person.sha256." + ("d" * 64)
_CLOSED_DOMAINS = frozenset({"fall", "bed_exit"})
_CLOSED_PRESENCE = frozenset({"PERSON_FOUND", "PERSON_GAP"})
_CLOSED_DUE = frozenset({"DUE", "NOT_DUE"})
_CLOSED_ALIGN = frozenset({"ALIGNED", "MISALIGNED"})


def _pose_components(state: str) -> tuple[tuple[str, str], ...]:
    return ((_POSE_COMPONENT, state), (_PERSON_COMPONENT, state))


def test_aligned_fall_latch_and_person_gap_facts_are_projected(tmp_path: Path) -> None:
    """Same-camera/boot/epoch/track/domain rows must expose latch and gap facts.

    Given an exact-30 fall FP with a same-track false-clear then new onset, a
    4-frame person gap, and mixed due/not-scheduled pose components
    When attribution evidence is extracted
    Then those persisted facts appear as typed evidence, never inferred zeros.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    gap_seqs = {36, 37, 38, 39}
    not_due_seqs = {20, 21}
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="fall-latch",
            seqs=seqs,
            analysis_track_by_seq={
                seq: None if seq in gap_seqs else 7 for seq in seqs
            },
            components_by_seq={
                seq: _pose_components(
                    "not-scheduled" if seq in not_due_seqs else "observed"
                )
                for seq in seqs
            },
            neighborhood_decisions=(
                {
                    "seq": 30,
                    "reason": "fall-onset",
                    "previous_state": "clear",
                    "current_state": "fall",
                    "triggered": 1,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.88, None),),
                },
                {
                    "seq": 33,
                    "reason": "below-threshold",
                    "previous_state": "fall",
                    "current_state": "clear",
                    "triggered": 0,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.12, None),),
                },
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)
    presence = record.person_presence
    due = record.due_signal
    latch = record.fall_latch
    alignment = record.domain_alignment

    assert {item.name for item in fields(record)} == _ALLOWED_EVIDENCE_FIELDS
    assert record.category is None
    assert record.evidence_status == "COMPLETE"
    assert record.coverage_status == "COMPLETE"
    assert record.expected_frames == NEIGHBORHOOD_SIZE
    assert record.retained_frames == NEIGHBORHOOD_SIZE
    assert record.worker_boot_id == BOOT_ID
    assert record.stream_epoch == EPOCH
    assert presence.status == "PERSON_GAP"
    assert presence.status in _CLOSED_PRESENCE
    assert presence.duration_frames == 4
    assert presence.missing_reason is None
    assert due.status == "NOT_DUE"
    assert due.status in _CLOSED_DUE
    assert due.not_scheduled_frames == 2
    assert due.missing_reason is None
    assert latch.same_track is True
    assert latch.same_domain is True
    assert latch.rise_before_rearm is True
    assert latch.rearm_frames == 7
    assert latch.missing_reason is None
    assert alignment.status == "ALIGNED"
    assert alignment.status in _CLOSED_ALIGN
    assert alignment.domain == "fall"
    assert alignment.domain in _CLOSED_DOMAINS
    assert alignment.same_track is True
    assert alignment.same_domain is True
    assert alignment.same_camera_boot_epoch is True
    assert alignment.missing_reason is None
    assert record.bed_state.status == "NOT_APPLICABLE"
    assert record.track_staleness.last_seen_offset_frames == 4
    assert record.track_staleness.missing_reason is None
    assert record.boot_changed is False
    assert record.epoch_changed is False
    assert record.boot_changed_missing_reason is None
    assert record.epoch_changed_missing_reason is None


def test_aligned_bed_stale_track_sequence_is_projected(tmp_path: Path) -> None:
    """Same-track bed state-machine rows must expose transition and staleness.

    Given an exact-30 bed-exit FP whose same-identity neighborhood contains
    contained -> live-grace -> stale-track-exit on track 7 / bed 2
    When attribution evidence is extracted
    Then the persisted sequence, durations, and last-seen offset are typed.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="bed-stale",
            seqs=seqs,
            reason="stale-track-exit",
            previous_state="live-grace",
            current_state="triggered",
            track_id=7,
            bed_id=2,
            module_qualified_id="bed_exit.v1",
            policy_qualified_id="bed_exit.policy.v1",
            event_type="bed-exit",
            values=(
                ("grace_frames_before", 1.0, None),
                ("grace_threshold", 3.0, None),
                ("containment_ratio", None, "track-no-longer-live"),
            ),
            analysis_track_by_seq={seq: None if seq >= 38 else 7 for seq in seqs},
            neighborhood_decisions=(
                {
                    "seq": 34,
                    "reason": "contained",
                    "previous_state": "contained",
                    "current_state": "contained",
                    "triggered": 0,
                    "track_id": 7,
                    "bed_id": 2,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("containment_ratio", 0.9, None),),
                },
                {
                    "seq": 37,
                    "reason": "live-grace",
                    "previous_state": "contained",
                    "current_state": "live-grace",
                    "triggered": 0,
                    "track_id": 7,
                    "bed_id": 2,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("grace_frames_before", 0.0, None),),
                },
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)
    bed = record.bed_state
    stale = record.track_staleness

    assert record.category is None
    assert record.evidence_status == "COMPLETE"
    assert record.decision_reason == "stale-track-exit"
    assert bed.status == "AVAILABLE"
    assert bed.sequence == ("contained", "live-grace", "triggered")
    assert bed.durations_frames == (3, 3, 1)
    assert bed.same_track is True
    assert bed.same_domain is True
    assert bed.missing_reason is None
    assert stale.last_seen_offset_frames == 2
    assert stale.same_track is True
    assert stale.missing_reason is None
    assert record.domain_alignment.domain == "bed_exit"
    assert record.domain_alignment.status == "ALIGNED"
    assert record.fall_latch.status == "NOT_APPLICABLE"


def test_misaligned_track_domain_and_identity_rows_are_not_proof(tmp_path: Path) -> None:
    """Foreign track/domain/camera/boot/epoch rows cannot complete domain proof.

    Given an exact-30 fall FP whose only rearm/gap/due rows live on another
    track, domain, camera, boot, or epoch
    When attribution evidence is extracted
    Then those required facts are typed missing/UNKNOWN, never false or zero.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="misaligned",
            seqs=seqs,
            extra_frames=(
                (OTHER_BOOT_ID, CAMERA_ID, EPOCH, 33),
                (BOOT_ID, OTHER_CAMERA_ID, EPOCH, 34),
                (BOOT_ID, CAMERA_ID, OTHER_EPOCH, 35),
            ),
            analysis_track_by_seq={seq: 7 for seq in seqs},
            components_by_seq={seq: _pose_components("observed") for seq in seqs},
            neighborhood_decisions=(
                {
                    "seq": 30,
                    "reason": "fall-onset",
                    "previous_state": "clear",
                    "current_state": "fall",
                    "triggered": 1,
                    "track_id": 8,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.8, None),),
                },
                {
                    "seq": 33,
                    "reason": "below-threshold",
                    "previous_state": "fall",
                    "current_state": "clear",
                    "triggered": 0,
                    "track_id": 7,
                    "boot_id": OTHER_BOOT_ID,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.1, None),),
                },
                {
                    "seq": 34,
                    "reason": "stale-track-exit",
                    "previous_state": "live-grace",
                    "current_state": "triggered",
                    "triggered": 1,
                    "track_id": 7,
                    "bed_id": 2,
                    "camera_id": OTHER_CAMERA_ID,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("grace_frames_before", 1.0, None),),
                },
                {
                    "seq": 35,
                    "reason": "stale-track-exit",
                    "previous_state": "live-grace",
                    "current_state": "triggered",
                    "triggered": 1,
                    "track_id": 7,
                    "bed_id": 2,
                    "epoch": OTHER_EPOCH,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("grace_frames_before", 1.0, None),),
                },
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.coverage_status == "COMPLETE"
    assert record.retained_frames == NEIGHBORHOOD_SIZE
    assert record.person_presence.status == "PERSON_FOUND"
    assert record.person_presence.duration_frames == 0
    assert record.due_signal.status == "DUE"
    assert record.due_signal.not_scheduled_frames == 0
    assert record.due_signal.missing_reason is None
    assert record.fall_latch.same_track is not True
    assert record.fall_latch.same_domain is not True
    assert record.fall_latch.rise_before_rearm is not True
    assert record.fall_latch.rearm_frames is None
    assert record.fall_latch.missing_reason is not None
    assert record.bed_state.status == "NOT_APPLICABLE"
    assert record.bed_state.sequence is None
    assert record.bed_state.durations_frames is None
    assert record.track_staleness.last_seen_offset_frames == 0
    assert record.domain_alignment.status is None
    assert record.domain_alignment.same_track is None
    assert record.domain_alignment.same_domain is None
    assert record.domain_alignment.same_camera_boot_epoch is None
    assert record.domain_alignment.missing_reason is not None
    assert record.boot_changed is not True
    assert record.epoch_changed is not True
    assert record.category is None


def test_missing_and_not_applicable_pose_states_are_not_due(tmp_path: Path) -> None:
    """Persisted missing/not-applicable pose states cannot become DUE/0.

    Given an exact-30 FP whose pose components are missing on one frame and
    not-applicable on another, with the rest observed
    When attribution evidence is extracted
    Then due_signal stays UNKNOWN-capable with a closed reason, never DUE/0.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="due-unknown",
            seqs=seqs,
            components_by_seq={
                seq: _pose_components(
                    "missing"
                    if seq == 20
                    else "not-applicable"
                    if seq == 21
                    else "observed"
                )
                for seq in seqs
            },
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.coverage_status == "COMPLETE"
    assert record.due_signal.status is None
    assert record.due_signal.not_scheduled_frames is None
    assert record.due_signal.missing_reason in {
        "observation_state_missing",
        "observation_state_not_applicable",
        "value_unparseable",
    }
    assert record.category is None


def test_trigger_seq_foreign_and_wrong_track_rows_are_not_aligned(
    tmp_path: Path,
) -> None:
    """Same-seq extras at the trigger cannot keep ALIGNED/true proof.

    Given an exact-30 fall FP plus foreign camera/boot/epoch and extra
    wrong-track decision rows on the trigger frame_seq
    When attribution evidence is extracted
    Then alignment stays None + closed reason, never ALIGNED/true.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="trigger-extra",
            seqs=seqs,
            extra_frames=(
                (OTHER_BOOT_ID, CAMERA_ID, EPOCH, TRIGGER_SEQ),
                (BOOT_ID, OTHER_CAMERA_ID, EPOCH, TRIGGER_SEQ),
                (BOOT_ID, CAMERA_ID, OTHER_EPOCH, TRIGGER_SEQ),
            ),
            extra_persons=((BOOT_ID, CAMERA_ID, EPOCH, TRIGGER_SEQ, 1, 8),),
            neighborhood_decisions=(
                {
                    "seq": TRIGGER_SEQ,
                    "reason": "below-threshold",
                    "previous_state": "fall",
                    "current_state": "clear",
                    "triggered": 0,
                    "track_id": 7,
                    "boot_id": OTHER_BOOT_ID,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.1, None),),
                },
                {
                    "seq": TRIGGER_SEQ,
                    "reason": "stale-track-exit",
                    "previous_state": "live-grace",
                    "current_state": "triggered",
                    "triggered": 1,
                    "track_id": 7,
                    "bed_id": 2,
                    "camera_id": OTHER_CAMERA_ID,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("grace_frames_before", 1.0, None),),
                },
                {
                    "seq": TRIGGER_SEQ,
                    "reason": "stale-track-exit",
                    "previous_state": "live-grace",
                    "current_state": "triggered",
                    "triggered": 1,
                    "track_id": 7,
                    "bed_id": 2,
                    "epoch": OTHER_EPOCH,
                    "module_qualified_id": "bed_exit.v1",
                    "policy_qualified_id": "bed_exit.policy.v1",
                    "values": (("grace_frames_before", 1.0, None),),
                },
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.coverage_status == "COMPLETE"
    assert record.retained_frames == NEIGHBORHOOD_SIZE
    assert record.domain_alignment.status is None
    assert record.domain_alignment.same_track is None
    assert record.domain_alignment.same_domain is None
    assert record.domain_alignment.same_camera_boot_epoch is None
    assert record.domain_alignment.missing_reason is not None
    assert record.fall_latch.same_track is None
    assert record.fall_latch.same_domain is None
    assert record.fall_latch.rise_before_rearm is None
    assert record.fall_latch.rearm_frames is None
    assert record.category is None


def test_incomplete_bed_sequence_does_not_infer_same_track_or_domain(
    tmp_path: Path,
) -> None:
    """A trigger-only bed row is not complete sequence proof.

    Given an exact-30 bed-exit FP with no preceding same-track/domain states
    When attribution evidence is extracted
    Then sequence/durations and same_track/same_domain stay None + reason.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="bed-incomplete",
            seqs=_complete_seqs(),
            reason="stale-track-exit",
            previous_state="live-grace",
            current_state="triggered",
            track_id=7,
            bed_id=2,
            module_qualified_id="bed_exit.v1",
            policy_qualified_id="bed_exit.policy.v1",
            event_type="bed-exit",
            values=(
                ("grace_frames_before", 1.0, None),
                ("grace_threshold", 3.0, None),
                ("containment_ratio", None, "track-no-longer-live"),
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.coverage_status == "COMPLETE"
    assert record.bed_state.status is None
    assert record.bed_state.sequence is None
    assert record.bed_state.durations_frames is None
    assert record.bed_state.same_track is None
    assert record.bed_state.same_domain is None
    assert record.bed_state.missing_reason is not None
    assert record.category is None


def test_person_gap_covers_leading_middle_and_trailing_null_tracks(
    tmp_path: Path,
) -> None:
    """Person-gap duration is the neighborhood null-track count, not trailing only.

    Given an exact-30 FP with leading, interior, and trailing null-track frames
    When attribution evidence is extracted
    Then PERSON_GAP duration_frames equals all those frames in frame units.
    """

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    gap_seqs = {11, 12, 20, 21, 22, 36, 37, 38, 39}
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="gap-patterns",
            seqs=seqs,
            analysis_track_by_seq={seq: None if seq in gap_seqs else 7 for seq in seqs},
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)

    assert record.coverage_status == "COMPLETE"
    assert record.person_presence.status == "PERSON_GAP"
    assert record.person_presence.duration_frames == 9
    assert record.person_presence.missing_reason is None
    assert record.boot_changed is False
    assert record.epoch_changed is False
    assert record.boot_changed_missing_reason is None
    assert record.epoch_changed_missing_reason is None


def test_absent_and_unparseable_domain_facts_stay_unknown_capable(tmp_path: Path) -> None:
    """Missing or hostile domain rows must stay UNKNOWN-capable, never inferred.

    Given an exact-30 FP with no neighborhood latch/bed rows and a hostile
    component observation_state
    When attribution evidence is extracted
    Then required domain facts are missing/UNKNOWN and no raw text leaks.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="unknown-domain",
            seqs=_complete_seqs(),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)
    rendered = repr(asdict(record))

    assert record.coverage_status == "COMPLETE"
    assert record.due_signal.status is None
    assert record.due_signal.not_scheduled_frames is None
    assert record.due_signal.missing_reason is not None
    assert record.fall_latch.missing_reason is not None
    assert record.fall_latch.rise_before_rearm is not False
    assert record.fall_latch.rearm_frames is None
    assert record.bed_state.sequence is None
    assert record.bed_state.durations_frames is None
    assert record.category is None
    assert PAYLOAD_SENTINEL not in rendered
    assert NOTE_SENTINEL not in rendered
    assert PATH_SENTINEL not in rendered

