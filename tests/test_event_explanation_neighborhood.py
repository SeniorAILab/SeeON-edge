from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

from backend.app.features.evidence.explanation_neighborhood import (
    EXPECTED_NEIGHBORHOOD_FRAMES,
    EventNeighborhoodQuery,
)
from shared.edge_db.migrator import migrate_database
from worker.pipeline.trace.store import TraceStore

MANIFEST_SHA256 = "a" * 64
POLICY_SHA256 = "b" * 64
NOW = "2026-08-13T00:00:00Z"
CAMERA_ID = "camera:alpha"
OTHER_CAMERA_ID = "camera:beta"
BOOT_ID = "boot:one"
OTHER_BOOT_ID = "boot:two"
EPOCH = 3
OTHER_EPOCH = 4
TRIGGER_SEQ = 40
PRECEDING_FRAMES = 29
NEIGHBORHOOD_SIZE = 30
DECISION_TRACE_ID = "c" * 64
EVENT_ID = "00000000-0000-4000-8000-000000000040"

_ALLOWED_COVERAGE_FIELDS = frozenset(
    {
        "neighborhood_pruned",
        "status",
        "coverage_reason",
        "expected_frames",
        "retained_frames",
        "first_missing_seq",
        "trigger",
        "cursor",
        "category",
        "prevented_eligible",
    }
)
_FORBIDDEN_COVERAGE_TOKENS = (
    "frames",
    "geometry",
    "polygon",
    "keypoints",
    "payload_json",
    "x1",
    "y1",
    "box",
    "coordinate",
)


def _trace_id(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _seed_manifest(
    connection: sqlite3.Connection,
    *,
    boot_id: str = BOOT_ID,
    camera_id: str = CAMERA_ID,
    applied_at: str = NOW,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_contents VALUES (?, 1, '{}', ?)",
        (MANIFEST_SHA256, applied_at),
    )
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_boots VALUES (?, ?, ?)",
        (boot_id, MANIFEST_SHA256, applied_at),
    )
    connection.execute(
        "INSERT OR IGNORE INTO runtime_manifest_cameras VALUES (?, ?, ?, ?)",
        (boot_id, camera_id, MANIFEST_SHA256, applied_at),
    )


def _insert_analysis(
    connection: sqlite3.Connection,
    *,
    seq: int,
    boot_id: str = BOOT_ID,
    camera_id: str = CAMERA_ID,
    epoch: int = EPOCH,
    source_time: float | None = None,
) -> str:
    trace_id = _trace_id(f"analysis:{boot_id}:{camera_id}:{epoch}:{seq}")
    time_value = float(seq) if source_time is None else source_time
    connection.execute(
        """
        INSERT INTO runtime_analysis_traces (
            trace_id, trace_schema_version, worker_boot_id, camera_id,
            stream_epoch, frame_seq, pts, source_time_sec, frame_width,
            frame_height, bed_region_provenance, storage_bytes
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, 16, 16, 'fresh', 1)
        """,
        (trace_id, boot_id, camera_id, epoch, seq, time_value, time_value),
    )
    return trace_id


def _insert_decision(
    connection: sqlite3.Connection,
    *,
    analysis_trace_id: str | None,
    decision_trace_id: str = DECISION_TRACE_ID,
) -> str:
    connection.execute(
        """
        INSERT INTO evidence_decision_traces (
            trace_id, trace_schema_version, analysis_trace_id,
            module_qualified_id, policy_qualified_id, effective_policy_id,
            runtime_manifest_sha256, reason, previous_state, current_state,
            triggered, track_id, track_missing_reason, bed_id, bed_missing_reason
        ) VALUES (?, 1, ?, 'fall.v1', 'fall.policy.v1', ?, ?, 'fall-onset',
                  'clear', 'triggered', 1, NULL, 'not-applicable', NULL,
                  'not-applicable')
        """,
        (decision_trace_id, analysis_trace_id, POLICY_SHA256, MANIFEST_SHA256),
    )
    return decision_trace_id


def _insert_event_ref(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str = EVENT_ID,
    decision_trace_id: str = DECISION_TRACE_ID,
) -> None:
    connection.execute(
        """
        INSERT INTO evidence_events (
            edge_event_id, detected_at, payload_json, state, queued_at,
            next_attempt_at, delivery_state
        ) VALUES (?, ?, '{"notes":"operator-private","x1":1}', 'ACKED', 1, 1, 'ACKED')
        """,
        (edge_event_id, NOW),
    )
    connection.execute(
        "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
        (edge_event_id, decision_trace_id),
    )


def _insert_cursor(
    connection: sqlite3.Connection,
    *,
    camera_id: str = CAMERA_ID,
    pruned_frames: int = 0,
    handoff_dropped_frames: int = 0,
    persistence_failed_frames: int = 0,
    retention_blocked_frames: int = 0,
    oldest_retained_seq: int | None = None,
    newest_retained_seq: int | None = None,
    oldest_boot_id: str | None = BOOT_ID,
    oldest_epoch: int | None = EPOCH,
    newest_boot_id: str | None = BOOT_ID,
    newest_epoch: int | None = EPOCH,
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
        ) VALUES (?, ?, ?, ?, ?, 40.0, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            handoff_dropped_frames,
            pruned_frames,
            oldest_retained_seq,
            newest_retained_seq,
            persistence_failed_frames,
            retention_blocked_frames,
            oldest_boot_id,
            oldest_epoch,
            oldest_trace,
            newest_boot_id,
            newest_epoch,
            newest_trace,
        ),
    )


def _seed_neighborhood(
    tmp_path: Path,
    *,
    seqs: tuple[int, ...],
    trigger_seq: int = TRIGGER_SEQ,
    extra_frames: tuple[tuple[str, str, int, int], ...] = (),
    cursor: dict[str, object] | None = None,
    include_decision: bool = True,
) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    connection = _connect(database)
    try:
        _seed_manifest(connection)
        _seed_manifest(connection, boot_id=OTHER_BOOT_ID, applied_at="2026-08-13T01:00:00Z")
        _seed_manifest(connection, camera_id=OTHER_CAMERA_ID)
        trigger_analysis_id: str | None = None
        for seq in seqs:
            analysis_id = _insert_analysis(connection, seq=seq)
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
        if include_decision:
            _insert_decision(connection, analysis_trace_id=trigger_analysis_id)
            _insert_event_ref(connection)
        if cursor is not None:
            _insert_cursor(connection, **cursor)
        connection.commit()
    finally:
        connection.close()
    return database


def _assert_counts_only(coverage: object) -> None:
    names = {item.name for item in fields(coverage)}
    assert names == _ALLOWED_COVERAGE_FIELDS
    serialized = repr(asdict(coverage))
    for token in _FORBIDDEN_COVERAGE_TOKENS:
        assert token not in names
        if token == "frames":
            continue
        assert token not in serialized


def test_current_analysis_timeline_recovers_across_boot_and_epoch_boundaries(
    tmp_path: Path,
) -> None:
    """Characterize today's camera-wide recovery: it is not neighborhood-bounded.

    Given the same camera has analysis rows on two boots and two epochs
    When recover_camera loads the persisted timeline
    Then every identity is returned in boot chronology, including foreign
    boot/epoch frames that a 30-frame neighborhood must never select.
    """

    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    connection = _connect(database)
    try:
        _seed_manifest(connection, boot_id=BOOT_ID, applied_at="2026-08-13T00:00:00Z")
        _seed_manifest(connection, boot_id=OTHER_BOOT_ID, applied_at="2026-08-13T01:00:00Z")
        for seq in (10, 11, 12):
            _insert_analysis(connection, seq=seq, boot_id=BOOT_ID, epoch=EPOCH)
        for seq in (0, 1, 2):
            _insert_analysis(connection, seq=seq, boot_id=BOOT_ID, epoch=OTHER_EPOCH)
        for seq in (0, 1):
            _insert_analysis(connection, seq=seq, boot_id=OTHER_BOOT_ID, epoch=1)
        connection.commit()
    finally:
        connection.close()

    recovered = TraceStore(database).recover_camera(CAMERA_ID)
    frame_keys = [frame.frame_key for frame in recovered.frames]

    assert frame_keys == [
        (BOOT_ID, CAMERA_ID, EPOCH, 10),
        (BOOT_ID, CAMERA_ID, EPOCH, 11),
        (BOOT_ID, CAMERA_ID, EPOCH, 12),
        (BOOT_ID, CAMERA_ID, OTHER_EPOCH, 0),
        (BOOT_ID, CAMERA_ID, OTHER_EPOCH, 1),
        (BOOT_ID, CAMERA_ID, OTHER_EPOCH, 2),
        (OTHER_BOOT_ID, CAMERA_ID, 1, 0),
        (OTHER_BOOT_ID, CAMERA_ID, 1, 1),
    ]
    assert {key[0] for key in frame_keys} == {BOOT_ID, OTHER_BOOT_ID}
    assert {key[2] for key in frame_keys} == {EPOCH, OTHER_EPOCH, 1}


def test_exact_30_contiguous_same_identity_rows_are_complete(tmp_path: Path) -> None:
    """Exact same-camera/boot/epoch 30-frame window is complete.

    Given the trigger and 29 preceding frame_seq rows share one identity
    When neighborhood coverage is queried from the decision trace
    Then the window is complete, unpruned, and eligible without a category.
    """

    seqs = tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1))
    extra = tuple(
        (OTHER_BOOT_ID, CAMERA_ID, EPOCH, seq) for seq in seqs
    ) + tuple((BOOT_ID, OTHER_CAMERA_ID, EPOCH, seq) for seq in seqs) + tuple(
        (BOOT_ID, CAMERA_ID, OTHER_EPOCH, seq) for seq in range(30)
    )
    database = _seed_neighborhood(tmp_path, seqs=seqs, extra_frames=extra)

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is False
    assert coverage.status == "COMPLETE"
    assert coverage.coverage_reason is None
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.expected_frames == EXPECTED_NEIGHBORHOOD_FRAMES
    assert coverage.retained_frames == NEIGHBORHOOD_SIZE
    assert coverage.first_missing_seq is None
    assert coverage.trigger is not None
    assert coverage.trigger.worker_boot_id == BOOT_ID
    assert coverage.trigger.camera_id == CAMERA_ID
    assert coverage.trigger.stream_epoch == EPOCH
    assert coverage.trigger.frame_seq == TRIGGER_SEQ
    assert coverage.category is None
    assert coverage.prevented_eligible is True
    _assert_counts_only(coverage)


def test_twenty_nine_same_identity_rows_are_neighborhood_pruned(tmp_path: Path) -> None:
    """A 29-row prefix of the window cannot become complete.

    Given only 29 contiguous same-identity rows including the trigger
    When neighborhood coverage is queried
    Then the window is pruned for an unexplained gap and stays ineligible.
    """

    seqs = tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1))
    database = _seed_neighborhood(tmp_path, seqs=seqs)

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "GAP"
    assert coverage.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == 29
    assert coverage.first_missing_seq == TRIGGER_SEQ - PRECEDING_FRAMES
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_fresh_epoch_prefix_is_pruned_and_ignores_prior_epoch(tmp_path: Path) -> None:
    """A new epoch shorter than 30 frames is prefix-short, not backfilled.

    Given trigger frame_seq is 10 and 0..10 exist, plus a prior epoch
    When neighborhood coverage is queried
    Then status is EPOCH_PREFIX_SHORT, the denominator stays 30, and no
    prior-epoch row is counted.
    """

    trigger_seq = 10
    database = _seed_neighborhood(
        tmp_path,
        seqs=tuple(range(trigger_seq + 1)),
        trigger_seq=trigger_seq,
        extra_frames=tuple((BOOT_ID, CAMERA_ID, OTHER_EPOCH, seq) for seq in range(20, 50)),
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "EPOCH_PREFIX_SHORT"
    assert coverage.coverage_reason == "NEIGHBORHOOD_EPOCH_PREFIX_SHORT"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.expected_frames >= NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == trigger_seq + 1
    assert coverage.first_missing_seq is None
    assert coverage.trigger is not None
    assert coverage.trigger.frame_seq == trigger_seq
    assert coverage.trigger.stream_epoch == EPOCH
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_missing_row_gap_is_pruned_without_selecting_foreign_fillers(
    tmp_path: Path,
) -> None:
    """An interior hole is an unexplained gap even if another identity has that seq.

    Given same-identity rows 11..40 except 25, and a foreign boot at seq 25
    When neighborhood coverage is queried
    Then the hole is GAP, retained stays 29, and the foreign row is unused.
    """

    seqs = tuple(seq for seq in range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1) if seq != 25)
    database = _seed_neighborhood(
        tmp_path,
        seqs=seqs,
        extra_frames=((OTHER_BOOT_ID, CAMERA_ID, EPOCH, 25),),
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "GAP"
    assert coverage.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == 29
    assert coverage.first_missing_seq == 25
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_pruned_trace_cursor_marks_retention_loss(tmp_path: Path) -> None:
    """Retention loss is typed PRUNED, not a silent short window.

    Given only seq 20..40 remain and the camera cursor reports pruned frames
    When neighborhood coverage is queried
    Then status is PRUNED with retained 21 and no category.
    """

    seqs = tuple(range(20, TRIGGER_SEQ + 1))
    database = _seed_neighborhood(
        tmp_path,
        seqs=seqs,
        cursor={
            "pruned_frames": 9,
            "oldest_retained_seq": 20,
            "newest_retained_seq": TRIGGER_SEQ,
        },
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "PRUNED"
    assert coverage.coverage_reason == "NEIGHBORHOOD_PRUNED"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == 21
    assert coverage.first_missing_seq == TRIGGER_SEQ - PRECEDING_FRAMES
    assert coverage.cursor is not None
    assert coverage.cursor.pruned_frames == 9
    assert coverage.cursor.oldest_retained_seq == 20
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_missing_trigger_analysis_is_pruned(tmp_path: Path) -> None:
    """A decision whose analysis row was unlinked cannot be classified.

    Given a valid decision whose analysis_trace_id is later set null
    When neighborhood coverage is queried
    Then the window is pruned for a missing trigger and stays ineligible.
    """

    database = _seed_neighborhood(
        tmp_path,
        seqs=tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1)),
    )
    connection = _connect(database)
    try:
        connection.execute(
            "UPDATE evidence_decision_traces SET analysis_trace_id = NULL "
            "WHERE trace_id = ?",
            (DECISION_TRACE_ID,),
        )
        connection.commit()
    finally:
        connection.close()

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "MISSING_TRIGGER"
    assert coverage.coverage_reason == "ANALYSIS_TRACE_NOT_RECORDED"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == 0
    assert coverage.trigger is None
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_unknown_decision_trace_is_missing_trigger(tmp_path: Path) -> None:
    """An absent decision trace is a missing trigger, not a zero window.

    Given a migrated database with no decision row
    When coverage is requested for an unknown decision id
    Then the result is pruned with DECISION_TRACE_NOT_RECORDED.
    """

    database = _seed_neighborhood(tmp_path, seqs=(), include_decision=False)

    coverage = EventNeighborhoodQuery(database).coverage_for_decision("f" * 64)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "MISSING_TRIGGER"
    assert coverage.coverage_reason == "DECISION_TRACE_NOT_RECORDED"
    assert coverage.expected_frames == NEIGHBORHOOD_SIZE
    assert coverage.retained_frames == 0
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_cross_camera_rows_cannot_complete_the_window(tmp_path: Path) -> None:
    """Another camera must never fill a missing predecessor.

    Given 29 same-camera rows and the missing seq on another camera
    When neighborhood coverage is queried
    Then the window stays pruned and does not report a boundary crossing.
    """

    seqs = tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1))
    database = _seed_neighborhood(
        tmp_path,
        seqs=seqs,
        extra_frames=((BOOT_ID, OTHER_CAMERA_ID, EPOCH, TRIGGER_SEQ - PRECEDING_FRAMES),),
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "GAP"
    assert coverage.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert coverage.retained_frames == 29
    assert coverage.first_missing_seq == TRIGGER_SEQ - PRECEDING_FRAMES
    assert coverage.trigger is not None
    assert coverage.trigger.camera_id == CAMERA_ID
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_cross_boot_rows_cannot_complete_the_window(tmp_path: Path) -> None:
    """Another worker boot must never fill a missing predecessor.

    Given 29 same-boot rows and the missing seq on another boot
    When neighborhood coverage is queried
    Then the foreign boot is unused and the window is pruned.
    """

    seqs = tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1))
    database = _seed_neighborhood(
        tmp_path,
        seqs=seqs,
        extra_frames=((OTHER_BOOT_ID, CAMERA_ID, EPOCH, TRIGGER_SEQ - PRECEDING_FRAMES),),
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "GAP"
    assert coverage.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert coverage.retained_frames == 29
    assert coverage.trigger is not None
    assert coverage.trigger.worker_boot_id == BOOT_ID
    assert coverage.category is None
    assert coverage.prevented_eligible is False


def test_cross_epoch_rows_cannot_complete_the_window(tmp_path: Path) -> None:
    """Another stream epoch must never fill a missing predecessor.

    Given 29 same-epoch rows and the missing seq on a prior epoch
    When neighborhood coverage is queried
    Then the prior epoch is unused and the window is pruned.
    """

    seqs = tuple(range(TRIGGER_SEQ - PRECEDING_FRAMES + 1, TRIGGER_SEQ + 1))
    database = _seed_neighborhood(
        tmp_path,
        seqs=seqs,
        extra_frames=((BOOT_ID, CAMERA_ID, OTHER_EPOCH, TRIGGER_SEQ - PRECEDING_FRAMES),),
    )

    coverage = EventNeighborhoodQuery(database).coverage_for_decision(DECISION_TRACE_ID)

    assert coverage.neighborhood_pruned is True
    assert coverage.status == "GAP"
    assert coverage.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert coverage.retained_frames == 29
    assert coverage.trigger is not None
    assert coverage.trigger.stream_epoch == EPOCH
    assert coverage.category is None
    assert coverage.prevented_eligible is False
