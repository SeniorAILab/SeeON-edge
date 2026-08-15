"""Exact same-camera/boot/epoch 30-frame neighborhood coverage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shared.edge_db.connection import RuntimeActor, open_runtime_database

EXPECTED_NEIGHBORHOOD_FRAMES = 30
_PRECEDING_FRAMES = EXPECTED_NEIGHBORHOOD_FRAMES - 1

NeighborhoodStatus = Literal[
    "COMPLETE",
    "EPOCH_PREFIX_SHORT",
    "PRUNED",
    "GAP",
    "CROSSED_BOUNDARY",
    "MISSING_TRIGGER",
]
CoverageReason = Literal[
    "NEIGHBORHOOD_PRUNED",
    "NEIGHBORHOOD_EPOCH_PREFIX_SHORT",
    "NEIGHBORHOOD_GAP_UNEXPLAINED",
    "NEIGHBORHOOD_CROSSES_BOOT_OR_EPOCH",
    "ANALYSIS_TRACE_NOT_RECORDED",
    "ANALYSIS_TRACE_DELETED_OR_UNLINKED",
    "DECISION_TRACE_NOT_RECORDED",
]


@dataclass(frozen=True, slots=True)
class NeighborhoodTrigger:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    frame_seq: int


@dataclass(frozen=True, slots=True)
class NeighborhoodCursor:
    pruned_frames: int
    handoff_dropped_frames: int
    persistence_failed_frames: int
    retention_blocked_frames: int
    oldest_retained_seq: int | None
    newest_retained_seq: int | None


@dataclass(frozen=True, slots=True)
class NeighborhoodCoverage:
    neighborhood_pruned: bool
    status: NeighborhoodStatus
    coverage_reason: CoverageReason | None
    expected_frames: int
    retained_frames: int
    first_missing_seq: int | None
    trigger: NeighborhoodTrigger | None
    cursor: NeighborhoodCursor | None
    category: None
    prevented_eligible: bool


def coverage_for_decision(
    connection: sqlite3.Connection,
    decision_trace_id: str,
) -> NeighborhoodCoverage:
    """Classify the trigger plus 29 preceding same-identity frames."""
    decision = connection.execute(
        "SELECT analysis_trace_id FROM evidence_decision_traces WHERE trace_id = ?",
        (decision_trace_id,),
    ).fetchone()
    if decision is None:
        return _missing("DECISION_TRACE_NOT_RECORDED")
    analysis_trace_id = decision[0]
    if analysis_trace_id is None:
        return _missing("ANALYSIS_TRACE_NOT_RECORDED")
    analysis = connection.execute(
        """
        SELECT worker_boot_id, camera_id, stream_epoch, frame_seq
        FROM runtime_analysis_traces
        WHERE trace_id = ?
        """,
        (str(analysis_trace_id),),
    ).fetchone()
    if analysis is None:
        return _missing("ANALYSIS_TRACE_DELETED_OR_UNLINKED")
    trigger = NeighborhoodTrigger(
        worker_boot_id=str(analysis[0]),
        camera_id=str(analysis[1]),
        stream_epoch=int(analysis[2]),
        frame_seq=int(analysis[3]),
    )
    cursor = _load_cursor(connection, trigger.camera_id)
    rows = connection.execute(
        """
        SELECT worker_boot_id, camera_id, stream_epoch, frame_seq
        FROM runtime_analysis_traces
        WHERE worker_boot_id = ?
          AND camera_id = ?
          AND stream_epoch = ?
          AND frame_seq >= ?
          AND frame_seq <= ?
        ORDER BY frame_seq
        """,
        (
            trigger.worker_boot_id,
            trigger.camera_id,
            trigger.stream_epoch,
            _window_start(trigger.frame_seq),
            trigger.frame_seq,
        ),
    ).fetchall()
    return _classify(trigger, rows, cursor)


class EventNeighborhoodQuery:
    """Count-only coverage for the trigger plus 29 preceding same-identity frames."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def coverage_for_decision(self, decision_trace_id: str) -> NeighborhoodCoverage:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            return coverage_for_decision(connection, decision_trace_id)
        finally:
            connection.close()


def _window_start(trigger_seq: int) -> int:
    start = trigger_seq - _PRECEDING_FRAMES
    return start if start > 0 else 0


def _expected_nonneg_seqs(trigger_seq: int) -> tuple[int, ...]:
    return tuple(range(_window_start(trigger_seq), trigger_seq + 1))


def _load_cursor(connection: sqlite3.Connection, camera_id: str) -> NeighborhoodCursor | None:
    row = connection.execute(
        """
        SELECT pruned_frames, handoff_dropped_frames, persistence_failed_frames,
               retention_blocked_frames, oldest_retained_seq, newest_retained_seq
        FROM runtime_trace_cursors
        WHERE camera_id = ?
        """,
        (camera_id,),
    ).fetchone()
    if row is None:
        return None
    return NeighborhoodCursor(
        pruned_frames=int(row[0]),
        handoff_dropped_frames=int(row[1]),
        persistence_failed_frames=int(row[2]),
        retention_blocked_frames=int(row[3]),
        oldest_retained_seq=None if row[4] is None else int(row[4]),
        newest_retained_seq=None if row[5] is None else int(row[5]),
    )


def _classify(
    trigger: NeighborhoodTrigger,
    rows: list[sqlite3.Row] | list[tuple[object, ...]],
    cursor: NeighborhoodCursor | None,
) -> NeighborhoodCoverage:
    for row in rows:
        if (
            str(row[0]) != trigger.worker_boot_id
            or str(row[1]) != trigger.camera_id
            or int(row[2]) != trigger.stream_epoch
        ):
            return NeighborhoodCoverage(
                neighborhood_pruned=True,
                status="CROSSED_BOUNDARY",
                coverage_reason="NEIGHBORHOOD_CROSSES_BOOT_OR_EPOCH",
                expected_frames=EXPECTED_NEIGHBORHOOD_FRAMES,
                retained_frames=0,
                first_missing_seq=None,
                trigger=trigger,
                cursor=cursor,
                category=None,
                prevented_eligible=False,
            )

    expected = _expected_nonneg_seqs(trigger.frame_seq)
    retained_seqs = tuple(int(row[3]) for row in rows)
    retained = frozenset(retained_seqs)
    missing = tuple(seq for seq in expected if seq not in retained)
    first_missing = None if not missing else missing[0]
    retained_count = len(retained)

    if trigger.frame_seq < _PRECEDING_FRAMES and not missing:
        return _incomplete(
            "EPOCH_PREFIX_SHORT",
            "NEIGHBORHOOD_EPOCH_PREFIX_SHORT",
            retained_count,
            None,
            trigger,
            cursor,
        )
    if missing and _retention_loss(cursor, expected[0] if expected else 0):
        return _incomplete(
            "PRUNED",
            "NEIGHBORHOOD_PRUNED",
            retained_count,
            first_missing,
            trigger,
            cursor,
        )
    if missing:
        return _incomplete(
            "GAP",
            "NEIGHBORHOOD_GAP_UNEXPLAINED",
            retained_count,
            first_missing,
            trigger,
            cursor,
        )
    if retained_count == EXPECTED_NEIGHBORHOOD_FRAMES:
        return NeighborhoodCoverage(
            neighborhood_pruned=False,
            status="COMPLETE",
            coverage_reason=None,
            expected_frames=EXPECTED_NEIGHBORHOOD_FRAMES,
            retained_frames=retained_count,
            first_missing_seq=None,
            trigger=trigger,
            cursor=cursor,
            category=None,
            prevented_eligible=True,
        )
    return _incomplete(
        "GAP",
        "NEIGHBORHOOD_GAP_UNEXPLAINED",
        retained_count,
        first_missing,
        trigger,
        cursor,
    )


def _retention_loss(cursor: NeighborhoodCursor | None, oldest_expected: int) -> bool:
    if cursor is None:
        return False
    if (
        cursor.pruned_frames > 0
        or cursor.handoff_dropped_frames > 0
        or cursor.persistence_failed_frames > 0
        or cursor.retention_blocked_frames > 0
    ):
        return True
    return (
        cursor.oldest_retained_seq is not None and cursor.oldest_retained_seq > oldest_expected
    )


def _incomplete(
    status: NeighborhoodStatus,
    reason: CoverageReason,
    retained_frames: int,
    first_missing_seq: int | None,
    trigger: NeighborhoodTrigger | None,
    cursor: NeighborhoodCursor | None,
) -> NeighborhoodCoverage:
    return NeighborhoodCoverage(
        neighborhood_pruned=True,
        status=status,
        coverage_reason=reason,
        expected_frames=EXPECTED_NEIGHBORHOOD_FRAMES,
        retained_frames=retained_frames,
        first_missing_seq=first_missing_seq,
        trigger=trigger,
        cursor=cursor,
        category=None,
        prevented_eligible=False,
    )


def _missing(reason: CoverageReason) -> NeighborhoodCoverage:
    return _incomplete("MISSING_TRIGGER", reason, 0, None, None, None)


__all__ = [
    "EXPECTED_NEIGHBORHOOD_FRAMES",
    "CoverageReason",
    "EventNeighborhoodQuery",
    "NeighborhoodCoverage",
    "NeighborhoodCursor",
    "NeighborhoodStatus",
    "NeighborhoodTrigger",
    "coverage_for_decision",
]
