from __future__ import annotations

import ast
import hashlib
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
        "epoch_changed",
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
        ) VALUES (?, 1, ?, 'fall.v1', 'fall.policy.v1', ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
        """,
        (
            decision_trace_id,
            analysis_trace_id,
            POLICY_ID,
            MANIFEST_ID,
            reason,
            previous_state,
            current_state,
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
) -> None:
    if decision_trace_id is None:
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, primary_clip_id, lifecycle_state,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'fall', ?, 'NOT_RECORDED', ?, 'STAGING', ?, ?)
            """,
            (incident_id, edge_event_id, camera_id, NOW, clip_id, NOW, NOW),
        )
        return
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            runtime_manifest_sha256, decision_trace_id, module_qualified_id,
            policy_qualified_id, effective_policy_id, provenance_state,
            primary_clip_id, lifecycle_state, created_at, updated_at
        ) VALUES (?, ?, ?, 'fall', ?, ?, ?, 'fall.v1', 'fall.policy.v1', ?,
                  'QUALIFIED', ?, 'STAGING', ?, ?)
        """,
        (
            incident_id,
            edge_event_id,
            camera_id,
            NOW,
            MANIFEST_ID,
            decision_trace_id,
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
    analysis_track_by_seq: dict[int, int] | None = None,
    disposition: str = "FALSE_POSITIVE",
    camera_id: str = CAMERA_ID,
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
        )
        if seq == trigger_seq:
            trigger_analysis_id = analysis_id
    for boot_id, camera_id, epoch, seq in extra_frames:
        _insert_analysis(
            connection,
            seq=seq,
            boot_id=boot_id,
            camera_id=camera_id,
            epoch=epoch,
        )
    resolved_decision = decision_trace_id or _trace_id(f"decision:{suffix}")
    if include_decision:
        _insert_decision(
            connection,
            decision_trace_id=resolved_decision,
            analysis_trace_id=trigger_analysis_id,
            track_id=track_id,
            track_missing_reason=None if track_id is not None else "not-applicable",
            bed_id=bed_id,
            bed_missing_reason=None if bed_id is not None else "not-applicable",
            values=() if values is None else values,
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
