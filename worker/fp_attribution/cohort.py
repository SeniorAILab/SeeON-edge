"""Read-only current false-positive cohort over schema v16 review state."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_CURRENT_SCHEMA_VERSION = 16
_LEFTOVER_EVENT_TYPE = "SYSTEM_TEST"
_FALSE_POSITIVE = "FALSE_POSITIVE"
_TRUE_POSITIVE = "TRUE_POSITIVE"
_MIGRATED = "MIGRATED"
_UNREVIEWED = "UNREVIEWED"
_UNMAPPABLE_LEGACY = "UNMAPPABLE_LEGACY"
_OPERATOR_ONLY_LEFTOVER = "OPERATOR_ONLY_LEFTOVER"
_TRACE_REF_CONFLICT = "TRACE_REF_CONFLICT"

_DDL_ACTION_NAMES = (
    "SQLITE_ALTER_TABLE",
    "SQLITE_ANALYZE",
    "SQLITE_ATTACH",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE",
    "SQLITE_DETACH",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_VTABLE",
    "SQLITE_REINDEX",
)
_DDL_ACTIONS = frozenset(
    getattr(sqlite3, name) for name in _DDL_ACTION_NAMES if hasattr(sqlite3, name)
)
_READ_PRAGMAS = frozenset(
    {
        "busy_timeout",
        "foreign_keys",
        "integrity_check",
        "journal_mode",
        "quick_check",
        "table_info",
        "table_xinfo",
        "user_version",
    }
)


@dataclass(frozen=True, slots=True)
class FalsePositiveCohortMember:
    edge_event_id: str
    incident_id: str
    current_review_version: int
    decision_trace_id: str | None


@dataclass(frozen=True, slots=True)
class FalsePositiveCohortExclusion:
    reason: str


@dataclass(frozen=True, slots=True)
class FalsePositiveCohort:
    members: tuple[FalsePositiveCohortMember, ...]
    exclusions: tuple[FalsePositiveCohortExclusion, ...]


class FalsePositiveCohortQuery:
    """Collapse current FP review state to one edge_event_id."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def load(self, connection: sqlite3.Connection | None = None) -> FalsePositiveCohort:
        if connection is not None:
            return _load_cohort(connection)
        owned = open_query_only_connection(self.database_path)
        try:
            return _load_cohort(owned)
        finally:
            owned.close()


def open_query_only_connection(database_path: Path) -> sqlite3.Connection:
    """Open an existing schema-v16 database for SELECT-only access."""
    path = _require_database_path(database_path)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        _configure_query_only(connection)
    except (OSError, sqlite3.Error, ValueError):
        connection.close()
        raise
    return connection


def _configure_query_only(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    version = connection.execute("PRAGMA user_version").fetchone()
    if version != (_CURRENT_SCHEMA_VERSION,):
        raise ValueError("edge-db schema is not v16")
    connection.set_authorizer(_query_only_authorizer())


def _load_cohort(connection: sqlite3.Connection) -> FalsePositiveCohort:
    members: dict[str, FalsePositiveCohortMember] = {}
    exclusions: list[FalsePositiveCohortExclusion] = []
    for row in connection.execute(_INCIDENT_SELECT):
        edge_event_id = _required_text(row[0])
        incident_id = _required_text(row[1])
        event_type = _required_text(row[2])
        review_version = row[3]
        disposition = _text(row[4])
        if event_type == _LEFTOVER_EVENT_TYPE:
            exclusions.append(FalsePositiveCohortExclusion(_OPERATOR_ONLY_LEFTOVER))
            continue
        if review_version is None or disposition is None:
            exclusions.append(FalsePositiveCohortExclusion(_UNREVIEWED))
            continue
        if disposition == _TRUE_POSITIVE:
            exclusions.append(FalsePositiveCohortExclusion(_TRUE_POSITIVE))
            continue
        if disposition != _FALSE_POSITIVE:
            exclusions.append(FalsePositiveCohortExclusion(disposition))
            continue
        resolved = _resolve_trace(_text(row[5]), _text(row[6]))
        if resolved == _TRACE_REF_CONFLICT:
            exclusions.append(FalsePositiveCohortExclusion(_TRACE_REF_CONFLICT))
            continue
        members[edge_event_id] = FalsePositiveCohortMember(
            edge_event_id=edge_event_id,
            incident_id=incident_id,
            current_review_version=_required_int(review_version),
            decision_trace_id=resolved,
        )
    for classification in connection.execute(_CLASSIFIED_LEGACY_SELECT):
        reason = _required_text(classification[0])
        if reason != _MIGRATED:
            exclusions.append(FalsePositiveCohortExclusion(reason))
    for _unused in connection.execute(_UNMAPPED_LABEL_SELECT):
        exclusions.append(FalsePositiveCohortExclusion(_UNMAPPABLE_LEGACY))
    return FalsePositiveCohort(
        members=tuple(members[key] for key in sorted(members)),
        exclusions=tuple(exclusions),
    )


def _resolve_trace(incident_trace_id: str | None, ref_trace_id: str | None) -> str | None:
    if (
        incident_trace_id is not None
        and ref_trace_id is not None
        and incident_trace_id != ref_trace_id
    ):
        return _TRACE_REF_CONFLICT
    if incident_trace_id is not None:
        return incident_trace_id
    return ref_trace_id


def _query_only_authorizer():
    def authorize(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        del database, source
        if action in _DDL_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            pragma = "" if argument_one is None else argument_one.lower()
            if pragma not in _READ_PRAGMAS or argument_two is not None:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return authorize


def _require_database_path(database_path: Path) -> Path:
    if not isinstance(database_path, Path) or not str(database_path) or not database_path.is_file():
        raise ValueError("edge-db is missing or unreadable")
    return database_path


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("stored text is invalid")
    return value


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored integer is invalid")
    return value


def _text(value: object) -> str | None:
    return None if value is None else _required_text(value)


_INCIDENT_SELECT = """
SELECT incident.edge_event_id, incident.incident_id, incident.event_type,
       review.review_version, review.disposition, incident.decision_trace_id,
       ref.decision_trace_id
FROM evidence_incidents AS incident
JOIN evidence_events AS event
  ON event.edge_event_id = incident.edge_event_id
LEFT JOIN control_evidence_review_state AS review_state
  ON review_state.incident_id = incident.incident_id
LEFT JOIN control_evidence_review_revisions AS review
  ON review.incident_id = review_state.incident_id
 AND review.clip_id = review_state.clip_id
 AND review.review_version = review_state.current_version
LEFT JOIN evidence_event_trace_refs AS ref
  ON ref.edge_event_id = incident.edge_event_id
ORDER BY incident.edge_event_id
"""

_CLASSIFIED_LEGACY_SELECT = """
SELECT classification
FROM control_legacy_label_migrations
WHERE classification != 'MIGRATED'
ORDER BY source_clip_id
"""

_UNMAPPED_LABEL_SELECT = """
SELECT clip_id
FROM labels
WHERE NOT EXISTS (
    SELECT 1
    FROM control_legacy_label_migrations AS classified
    WHERE classified.source_clip_id = labels.clip_id
)
ORDER BY clip_id
"""

__all__ = [
    "FalsePositiveCohort",
    "FalsePositiveCohortExclusion",
    "FalsePositiveCohortMember",
    "FalsePositiveCohortQuery",
    "open_query_only_connection",
]
