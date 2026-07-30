"""Atomic immutable staging for evidence outbox events."""

from __future__ import annotations

import sqlite3
from types import TracebackType
from typing import Literal, Self

from edge.evidence.evidence_outbox_types import StagedEvent, StagedEventConflictError


def stage_event(connection: sqlite3.Connection, event: StagedEvent) -> None:
    """Insert once, accepting only exact immutable replays of a stable event ID."""
    with _ImmediateTransaction(connection):
        existing = connection.execute(
            """
            SELECT detected_at, payload_json
            FROM evidence_events
            WHERE edge_event_id = ?
            """,
            (event.edge_event_id,),
        ).fetchone()
        if existing is not None:
            _refuse_mismatched_replay(event, existing)
            return
        connection.execute(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state,
                queued_at, next_attempt_at
            ) VALUES (?, ?, ?, 'STAGED', ?, ?)
            """,
            (
                event.edge_event_id,
                event.detected_at,
                event.payload_json,
                event.queued_at,
                event.queued_at,
            ),
        )


def _refuse_mismatched_replay(event: StagedEvent, existing: sqlite3.Row) -> None:
    mismatches: list[Literal["detected_at", "payload_json"]] = []
    if str(existing[0]) != event.detected_at:
        mismatches.append("detected_at")
    if str(existing[1]) != event.payload_json:
        mismatches.append("payload_json")
    if mismatches:
        raise StagedEventConflictError(event.edge_event_id, tuple(mismatches))


class _ImmediateTransaction:
    """Rollback-capable guard for the atomic compare-or-insert operation."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> Self:
        self._connection.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self._connection.commit()
            return
        self._connection.rollback()


__all__ = ["stage_event"]
