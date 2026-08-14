"""Durable PREPARED -> DB_COMMITTED -> DONE apply journal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_database import RelationPlan
from worker.pipeline.output.evidence.clip_consistency_io import (
    atomic_write_json,
    read_strict_json,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_journal_validation import (
    JOURNAL_KEYS,
    counters,
    insert_rows,
    integer,
    journal_state,
    optional_string,
    quarantine,
    string,
    strings,
    validate_counter_facts,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    FaultHook,
    JournalState,
    RepairCounters,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ApplyJournal:
    state: JournalState
    owner_uid: int
    state_db: str
    clip_store: str
    journal_path: str
    source_state_sha256: str
    plan_sha256: str
    relations_before_sha256: str
    relations_after_sha256: str
    backup_receipt_path: str
    backup_receipt_sha256: str
    delete_event_ids: tuple[str, ...]
    insert_rows: tuple[tuple[str, str, int], ...]
    quarantine: tuple[tuple[str, str], ...]
    counters: RepairCounters
    error: str | None = None

    def relation_plan(self) -> RelationPlan:
        return RelationPlan(
            relations_before=self.counters.relations_before,
            relations_after=self.counters.relations_after,
            mismatch_clips=self.counters.mismatch_clips,
            mismatch_tuples=self.counters.mismatch_tuples,
            delete_event_ids=self.delete_event_ids,
            insert_rows=self.insert_rows,
            before_sha256=self.relations_before_sha256,
            after_sha256=self.relations_after_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "state": self.state,
            "schema_version": SCHEMA_VERSION,
            "owner_uid": self.owner_uid,
            "state_db": self.state_db,
            "clip_store": self.clip_store,
            "journal_path": self.journal_path,
            "source_state_sha256": self.source_state_sha256,
            "plan_sha256": self.plan_sha256,
            "relations_before_sha256": self.relations_before_sha256,
            "relations_after_sha256": self.relations_after_sha256,
            "backup_receipt_path": self.backup_receipt_path,
            "backup_receipt_sha256": self.backup_receipt_sha256,
            "delete_event_ids": list(self.delete_event_ids),
            "insert_rows": [list(row) for row in self.insert_rows],
            "quarantine": [list(row) for row in self.quarantine],
            "counters": asdict(self.counters),
            "error": self.error,
        }


def prepared_journal(
    *,
    owner_uid: int,
    state_db: Path,
    clip_store: Path,
    journal_path: Path,
    source_state_sha256: str,
    backup_receipt_path: str,
    backup_receipt_sha256: str,
    plan: RelationPlan,
    quarantine: tuple[tuple[str, str], ...],
    counters: RepairCounters,
) -> ApplyJournal:
    return ApplyJournal(
        state="PREPARED",
        owner_uid=owner_uid,
        state_db=str(state_db.absolute()),
        clip_store=str(clip_store.absolute()),
        journal_path=str(journal_path.absolute()),
        source_state_sha256=source_state_sha256,
        plan_sha256=plan.plan_sha256,
        relations_before_sha256=plan.before_sha256,
        relations_after_sha256=plan.after_sha256,
        backup_receipt_path=backup_receipt_path,
        backup_receipt_sha256=backup_receipt_sha256,
        delete_event_ids=plan.delete_event_ids,
        insert_rows=plan.insert_rows,
        quarantine=quarantine,
        counters=counters,
    )


def write_journal(
    journal: ApplyJournal,
    *,
    path: Path,
    maintenance_root: Path,
    expected_uid: int,
    hook: FaultHook | None,
    stage: str,
) -> None:
    atomic_write_json(
        path,
        journal.to_dict(),
        root=maintenance_root,
        expected_uid=expected_uid,
        hook=hook,
        stage=stage,
    )


def mark_journal(
    journal: ApplyJournal,
    state: JournalState,
    *,
    path: Path,
    maintenance_root: Path,
    expected_uid: int,
    hook: FaultHook | None,
    error: str | None = None,
) -> ApplyJournal:
    updated = replace(journal, state=state, error=error)
    write_journal(
        updated,
        path=path,
        maintenance_root=maintenance_root,
        expected_uid=expected_uid,
        hook=hook,
        stage=f"journal_{state.lower()}",
    )
    return updated


def read_journal(
    path: Path,
    *,
    maintenance_root: Path,
    expected_uid: int,
    state_db: Path,
    clip_store: Path,
) -> ApplyJournal:
    validate_under_root(path, maintenance_root, allow_missing_leaf=False)
    payload = read_strict_json(
        path,
        expected_uid=expected_uid,
        exact_mode=0o600,
        error_code="journal_invalid",
    )
    if frozenset(payload) != JOURNAL_KEYS:
        raise ClipConsistencyError("journal_invalid", "journal key set differs")
    journal = ApplyJournal(
        state=journal_state(string(payload, "state")),
        owner_uid=integer(payload, "owner_uid"),
        state_db=string(payload, "state_db"),
        clip_store=string(payload, "clip_store"),
        journal_path=string(payload, "journal_path"),
        source_state_sha256=string(payload, "source_state_sha256"),
        plan_sha256=string(payload, "plan_sha256"),
        relations_before_sha256=string(payload, "relations_before_sha256"),
        relations_after_sha256=string(payload, "relations_after_sha256"),
        backup_receipt_path=string(payload, "backup_receipt_path"),
        backup_receipt_sha256=string(payload, "backup_receipt_sha256"),
        delete_event_ids=strings(payload.get("delete_event_ids")),
        insert_rows=insert_rows(payload.get("insert_rows")),
        quarantine=quarantine(payload.get("quarantine")),
        counters=counters(payload.get("counters")),
        error=optional_string(payload, "error"),
    )
    validate_counter_facts(
        journal.state,
        journal.counters,
        deletes=len(journal.delete_event_ids),
        inserts=len(journal.insert_rows),
        quarantines=len(journal.quarantine),
    )
    valid_identity = (
        integer(payload, "format_version") == _FORMAT_VERSION
        and integer(payload, "schema_version") == SCHEMA_VERSION
        and journal.owner_uid == expected_uid
        and journal.state_db == str(state_db.absolute())
        and journal.clip_store == str(clip_store.absolute())
        and journal.journal_path == str(path.absolute())
        and journal.relation_plan().plan_sha256 == journal.plan_sha256
    )
    if not valid_identity:
        raise ClipConsistencyError("journal_invalid", "journal identity differs")
    return journal


__all__ = [
    "ApplyJournal",
    "mark_journal",
    "prepared_journal",
    "read_journal",
    "write_journal",
]
