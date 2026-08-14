"""Schema-9 database validation and relation planning for clip repair."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

_EXPECTED_COLUMNS: Final = {
    "evidence_events": (
        "edge_event_id", "detected_at", "payload_json", "state", "queued_at",
        "next_attempt_at", "attempt_count", "lease_owner", "lease_expires_at",
        "delivery_state", "backend_event_id", "last_error_code",
    ),
    "evidence_clips": (
        "clip_id", "local_state", "manifest_path", "state_version", "media_relpath",
        "sha256", "size_bytes", "mime_type", "codec", "duration_ms",
        "clip_start_at", "clip_end_at", "finalized_at", "unavailable_reason",
        "publish_state", "publish_attempt_count", "publish_next_attempt_at",
        "publish_lease_owner", "publish_lease_expires_at", "remote_state",
        "backend_ack_at", "last_error_code",
    ),
    "clip_events": ("clip_id", "edge_event_id", "ordinal"),
}


@dataclass(frozen=True, slots=True)
class RelationPlan:
    relations_before: int
    relations_after: int
    delete_event_ids: tuple[str, ...]
    insert_rows: tuple[tuple[str, str, int], ...]


def validate_database(connection: sqlite3.Connection, *, now: float) -> None:
    version_row = connection.execute("PRAGMA user_version").fetchone()
    if version_row is None or int(version_row[0]) != SCHEMA_VERSION:
        raise ClipConsistencyError("schema_drift", "database is not schema 9")
    for table, expected in _EXPECTED_COLUMNS.items():
        columns = tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))
        if columns != expected:
            raise ClipConsistencyError("schema_drift", f"{table} columns differ")
        table_row = connection.execute(
            "SELECT strict FROM pragma_table_list WHERE name = ? AND schema = 'main'",
            (table,),
        ).fetchone()
        if table_row is None or int(table_row[0]) != 1:
            raise ClipConsistencyError("schema_drift", f"{table} is not strict")
    foreign_keys = {
        (str(row[2]), str(row[3]), str(row[4]), str(row[6]))
        for row in connection.execute("PRAGMA foreign_key_list(clip_events)")
    }
    expected_foreign_keys = {
        ("evidence_clips", "clip_id", "clip_id", "RESTRICT"),
        ("evidence_events", "edge_event_id", "edge_event_id", "RESTRICT"),
    }
    if foreign_keys != expected_foreign_keys:
        raise ClipConsistencyError("schema_drift", "clip_events foreign keys differ")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ClipConsistencyError("integrity_drift", "database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "database foreign keys are invalid")
    active_event = connection.execute(
        "SELECT 1 FROM evidence_events WHERE state = 'IN_FLIGHT' AND lease_expires_at > ? LIMIT 1",
        (now,),
    ).fetchone()
    active_clip = connection.execute(
        "SELECT 1 FROM evidence_clips WHERE publish_state = 'IN_FLIGHT' "
        "AND publish_lease_expires_at > ? LIMIT 1",
        (now,),
    ).fetchone()
    if active_event is not None or active_clip is not None:
        raise ClipConsistencyError("active_lease", "active evidence lease exists")


def plan_relations(
    connection: sqlite3.Connection,
    desired: dict[str, tuple[str, ...]],
) -> RelationPlan:
    clip_rows = {
        str(row[0]): str(row[1])
        for row in connection.execute("SELECT clip_id, local_state FROM evidence_clips")
    }
    event_ids = {
        str(row[0])
        for row in connection.execute("SELECT edge_event_id FROM evidence_events")
    }
    desired_refs = [event_id for refs in desired.values() for event_id in refs]
    if len(desired_refs) != len(set(desired_refs)):
        raise ClipConsistencyError("manifest_conflict", "event appears in multiple final manifests")
    for clip_id, refs in desired.items():
        if clip_id not in clip_rows:
            raise ClipConsistencyError("database_drift", "final clip is absent from evidence_clips")
        if not set(refs) <= event_ids:
            raise ClipConsistencyError(
                "database_drift", "manifest event is absent from evidence_events"
            )

    existing_by_clip: dict[str, tuple[tuple[str, int], ...]] = {}
    for clip_id in desired:
        rows = connection.execute(
            "SELECT edge_event_id, ordinal FROM clip_events "
            "WHERE clip_id = ? ORDER BY ordinal",
            (clip_id,),
        ).fetchall()
        existing_by_clip[clip_id] = tuple((str(row[0]), int(row[1])) for row in rows)
    changed = {
        clip_id
        for clip_id, refs in desired.items()
        if existing_by_clip[clip_id]
        != tuple((event_id, ordinal) for ordinal, event_id in enumerate(refs))
    }
    delete_ids = {
        str(row[0])
        for clip_id in changed
        for row in connection.execute(
            "SELECT edge_event_id FROM clip_events WHERE clip_id = ?", (clip_id,)
        )
    }
    if desired_refs:
        placeholders = ",".join("?" for _ in desired_refs)
        delete_ids.update(
            str(row[0])
            for row in connection.execute(
                f"SELECT edge_event_id FROM clip_events WHERE edge_event_id IN ({placeholders}) "
                f"AND clip_id NOT IN ({','.join('?' for _ in desired)})",
                (*desired_refs, *desired),
            )
        )
    inserts = tuple(
        (clip_id, event_id, ordinal)
        for clip_id in sorted(changed)
        for ordinal, event_id in enumerate(desired[clip_id])
    )
    before = int(connection.execute("SELECT COUNT(*) FROM clip_events").fetchone()[0])
    return RelationPlan(
        before,
        before - len(delete_ids) + len(inserts),
        tuple(sorted(delete_ids)),
        inserts,
    )


__all__ = ["RelationPlan", "plan_relations", "validate_database"]
