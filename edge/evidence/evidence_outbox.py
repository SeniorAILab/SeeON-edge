"""Durable worker-owned evidence outbox."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

from edge.evidence import evidence_outbox_clips as clip_store
from edge.evidence import evidence_outbox_delivery as delivery_store
from edge.evidence.evidence_outbox_schema import MIGRATIONS, SCHEMA_VERSION
from edge.evidence.evidence_outbox_stage import stage_event
from edge.evidence.evidence_outbox_types import (
    ClaimedClip,
    ClaimedEvent,
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    DatabaseSettings,
    EdgeEventId,
    EventClipConflictError,
    EvidenceReasonCode,
    MissingStagedEventError,
    NewerSchemaVersionError,
    StagedEvent,
    StagedEventConflictError,
)


class EvidenceOutbox:
    """Stateful SQLite resource for lossless worker evidence delivery."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path) -> Self:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
        version_row = connection.execute("PRAGMA user_version").fetchone()
        version = 0 if version_row is None else int(version_row[0])
        if version > SCHEMA_VERSION:
            connection.close()
            raise NewerSchemaVersionError(found=version, supported=SCHEMA_VERSION)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA journal_mode = WAL")
            _migrate(connection, version)
        except sqlite3.Error:
            connection.close()
            raise
        return cls(connection)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        self._connection.close()

    def stage(self, event: StagedEvent) -> None:
        stage_event(self._connection, event)

    def bind_clip(self, edge_event_id: EdgeEventId, clip_id: ClipId) -> int:
        return clip_store.bind_clip(self._connection, edge_event_id, clip_id)

    def mark_ready(self, edge_event_id: EdgeEventId) -> bool:
        result = self._connection.execute(
            """
            UPDATE evidence_events
            SET state = 'READY'
            WHERE edge_event_id = ? AND state = 'STAGED'
            """,
            (edge_event_id,),
        )
        return result.rowcount == 1

    def claim(self, lease: ClaimLease) -> ClaimedEvent | None:
        with _ImmediateTransaction(self._connection):
            candidate = self._connection.execute(
                """
                SELECT edge_event_id
                FROM evidence_events
                WHERE next_attempt_at <= ?
                  AND delivery_state = 'PENDING'
                  AND (
                    state = 'READY'
                    OR (state = 'IN_FLIGHT' AND lease_expires_at <= ?)
                  )
                ORDER BY queued_at, edge_event_id
                LIMIT 1
                """,
                (lease.now, lease.now),
            ).fetchone()
            if candidate is None:
                return None
            edge_event_id = EdgeEventId(str(candidate[0]))
            lease_expires_at = lease.now + lease.duration
            self._connection.execute(
                """
                UPDATE evidence_events
                SET state = 'IN_FLIGHT', lease_owner = ?, lease_expires_at = ?,
                    attempt_count = attempt_count + 1
                WHERE edge_event_id = ?
                """,
                (lease.owner, lease_expires_at, edge_event_id),
            )
            claimed = self._connection.execute(
                """
                SELECT event.edge_event_id, event.detected_at, event.payload_json,
                       relation.clip_id, event.attempt_count
                FROM evidence_events AS event
                LEFT JOIN clip_events AS relation USING (edge_event_id)
                WHERE event.edge_event_id = ?
                """,
                (edge_event_id,),
            ).fetchone()
            if claimed is None:
                raise MissingStagedEventError(edge_event_id)
        clip_id = None if claimed[3] is None else ClipId(str(claimed[3]))
        return ClaimedEvent(
            edge_event_id=EdgeEventId(str(claimed[0])),
            detected_at=str(claimed[1]),
            payload_json=str(claimed[2]),
            clip_id=clip_id,
            lease_owner=lease.owner,
            lease_expires_at=lease_expires_at,
            attempt_count=int(claimed[4]),
        )

    def schedule_retry(self, claim: ClaimedEvent, *, next_attempt_at: float) -> bool:
        result = self._connection.execute(
            """
            UPDATE evidence_events
            SET state = 'READY', next_attempt_at = ?,
                lease_owner = NULL, lease_expires_at = NULL
            WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
              AND lease_owner = ? AND lease_expires_at = ?
            """,
            (
                next_attempt_at,
                claim.edge_event_id,
                claim.lease_owner,
                claim.lease_expires_at,
            ),
        )
        return result.rowcount == 1

    def acknowledge(self, claim: ClaimedEvent, *, backend_event_id: str | None = None) -> bool:
        result = self._connection.execute(
            """
            UPDATE evidence_events
            SET state = 'ACKED', delivery_state = 'ACKED', backend_event_id = ?,
                lease_owner = NULL, lease_expires_at = NULL, last_error_code = NULL
            WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
              AND lease_owner = ? AND lease_expires_at = ?
            """,
            (
                backend_event_id,
                claim.edge_event_id,
                claim.lease_owner,
                claim.lease_expires_at,
            ),
        )
        return result.rowcount == 1

    def mark_event_failure(self, claim: ClaimedEvent, *, state: str, error_code: str) -> bool:
        result = self._connection.execute(
            """
            UPDATE evidence_events
            SET state = 'READY', delivery_state = ?, last_error_code = ?,
                lease_owner = NULL, lease_expires_at = NULL
            WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
              AND lease_owner = ? AND lease_expires_at = ?
            """,
            (state, error_code, claim.edge_event_id, claim.lease_owner, claim.lease_expires_at),
        )
        return result.rowcount == 1

    def event_delivery_state(self, edge_event_id: EdgeEventId) -> str | None:
        row = self._connection.execute(
            "SELECT delivery_state FROM evidence_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone()
        return None if row is None else str(row[0])

    def event_attempt_count(self, edge_event_id: EdgeEventId) -> int | None:
        row = self._connection.execute(
            "SELECT attempt_count FROM evidence_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone()
        return None if row is None else int(row[0])

    def claim_clip(self, lease: ClaimLease) -> ClaimedClip | None:
        return delivery_store.claim_clip(self._connection, lease)

    def schedule_clip_retry(
        self,
        claim: ClaimedClip,
        *,
        next_attempt_at: float,
        error_code: str,
    ) -> bool:
        return delivery_store.schedule_clip_retry(
            self._connection,
            claim,
            next_attempt_at=next_attempt_at,
            error_code=error_code,
        )

    def acknowledge_clip(
        self,
        claim: ClaimedClip,
        *,
        acknowledged_at: float,
        remote_state: str,
    ) -> bool:
        return delivery_store.acknowledge_clip(
            self._connection,
            claim,
            acknowledged_at=acknowledged_at,
            remote_state=remote_state,
        )

    def mark_clip_failure(self, claim: ClaimedClip, *, state: str, error_code: str) -> bool:
        return delivery_store.mark_clip_failure(
            self._connection, claim, state=state, error_code=error_code
        )

    def clip_publish_state(self, clip_id: ClipId) -> str | None:
        return delivery_store.clip_publish_state(self._connection, clip_id)

    def is_clip_held(self, clip_id: ClipId) -> bool:
        return delivery_store.is_clip_held(self._connection, clip_id)

    def clip_remote_state(self, clip_id: ClipId) -> str | None:
        row = self._connection.execute(
            "SELECT remote_state FROM evidence_clips WHERE clip_id = ?", (clip_id,)
        ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def held_clip_ids(self) -> tuple[ClipId, ...]:
        rows = self._connection.execute(
            """
            SELECT clip_id FROM evidence_clips
            WHERE publish_state != 'PUBLISHED' ORDER BY clip_id
            """
        ).fetchall()
        return tuple(ClipId(str(row[0])) for row in rows)

    def release_compatibility(self) -> None:
        delivery_store.release_compatibility(self._connection)

    def pending_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM evidence_events WHERE state != 'ACKED'"
        ).fetchone()
        return 0 if row is None else int(row[0])

    def ordered_event_ids(self, clip_id: ClipId) -> tuple[EdgeEventId, ...]:
        return clip_store.ordered_event_ids(self._connection, clip_id)

    def awaiting_clip_ids(self) -> tuple[ClipId, ...]:
        return clip_store.awaiting_clip_ids(self._connection)

    def record_clip_outcome(self, outcome: ClipOutcome) -> None:
        clip_store.record_clip_outcome(self._connection, outcome)

    def reconcile_clip(
        self,
        event_ids: tuple[EdgeEventId, ...],
        outcome: ClipOutcome,
    ) -> None:
        clip_store.reconcile_clip(self._connection, event_ids, outcome)

    def clip_outcome(self, clip_id: ClipId) -> ClipOutcome | None:
        return clip_store.clip_outcome(self._connection, clip_id)

    def database_settings(self) -> DatabaseSettings:
        journal = self._connection.execute("PRAGMA journal_mode").fetchone()
        synchronous = self._connection.execute("PRAGMA synchronous").fetchone()
        foreign_keys = self._connection.execute("PRAGMA foreign_keys").fetchone()
        return DatabaseSettings(
            journal_mode="" if journal is None else str(journal[0]),
            synchronous=0 if synchronous is None else int(synchronous[0]),
            foreign_keys=0 if foreign_keys is None else int(foreign_keys[0]),
        )


class _ImmediateTransaction:
    """Rollback-capable guard for one explicit SQLite write transaction."""

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


def _migrate(connection: sqlite3.Connection, version: int) -> None:
    if version == SCHEMA_VERSION:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            for statement in MIGRATIONS[target_version - 1]:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {target_version}")
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise


__all__ = [
    "ClaimLease",
    "ClaimedClip",
    "ClaimedEvent",
    "ClipId",
    "ClipLocalState",
    "ClipOutcome",
    "ClipOutcomeConflictError",
    "DatabaseSettings",
    "EdgeEventId",
    "EvidenceReasonCode",
    "EventClipConflictError",
    "EvidenceOutbox",
    "MissingStagedEventError",
    "NewerSchemaVersionError",
    "StagedEvent",
    "StagedEventConflictError",
]
