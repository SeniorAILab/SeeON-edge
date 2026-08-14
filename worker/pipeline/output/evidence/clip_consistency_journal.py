"""Durable PREPARED -> DB_COMMITTED -> DONE apply journal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_authority import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_database import RelationPlan
from worker.pipeline.output.evidence.clip_consistency_io import (
    read_strict_json,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_journal_io import write_journal
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
from worker.pipeline.output.evidence.clip_consistency_operation import OperationBinding
from worker.pipeline.output.evidence.clip_consistency_phase_authority import (
    AuthoritySnapshot,
    FileIdentity,
    ProofIdentity,
    validate_journal_identity,
)
from worker.pipeline.output.evidence.clip_consistency_quarantine import (
    canonical_quarantine_rows,
    quarantine_rows_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    FaultHook,
    JournalState,
    RepairCounters,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

_FORMAT_VERSION = 4


@dataclass(frozen=True, slots=True)
class ApplyJournal:
    state: JournalState
    state_uid: int
    state_gid: int
    state_db_mode: int
    state_dir_mode: int
    clip_uid: int
    clip_gid: int
    clip_dir_mode: int
    tool_revision: str
    authority_sha256: str
    operation_digest_version: int
    operation_digest: str
    quarantine_namespace_sha256: str
    image_artifact_identity: str
    state_db: str
    clip_store: str
    maintenance_root: str
    journal_path: str
    proof_identity: ProofIdentity
    authority_snapshot: AuthoritySnapshot
    source_state_sha256: str
    source_identity_sha256: str
    non_relation_state_sha256: str
    plan_sha256: str
    relations_before_sha256: str
    relations_after_sha256: str
    backup_receipt_path: str
    backup_receipt_sha256: str
    delete_event_ids: tuple[str, ...]
    insert_rows: tuple[tuple[str, str, int], ...]
    quarantine_clip_ids: tuple[str, ...]
    quarantine: tuple[tuple[str, str], ...]
    quarantine_sha256: str
    deleted_quarantine: tuple[str, ...]
    counters: RepairCounters
    error: str | None = None
    file_identity: FileIdentity | None = field(default=None, compare=False, repr=False)

    def relation_plan(self) -> RelationPlan:
        return RelationPlan(
            relations_before=self.counters.relations_before,
            relations_after=self.counters.relations_after,
            mismatch_clips=self.counters.mismatch_clips,
            mismatch_tuples=self.counters.mismatch_tuples,
            delete_event_ids=self.delete_event_ids,
            insert_rows=self.insert_rows,
            quarantine_clip_ids=self.quarantine_clip_ids,
            before_sha256=self.relations_before_sha256,
            after_sha256=self.relations_after_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "state": self.state,
            "schema_version": SCHEMA_VERSION,
            "state_uid": self.state_uid,
            "state_gid": self.state_gid,
            "state_db_mode": self.state_db_mode,
            "state_dir_mode": self.state_dir_mode,
            "clip_uid": self.clip_uid,
            "clip_gid": self.clip_gid,
            "clip_dir_mode": self.clip_dir_mode,
            "tool_revision": self.tool_revision,
            "authority_sha256": self.authority_sha256,
            "operation_digest_version": self.operation_digest_version,
            "operation_digest": self.operation_digest,
            "quarantine_namespace_sha256": self.quarantine_namespace_sha256,
            "image_artifact_identity": self.image_artifact_identity,
            "state_db": self.state_db,
            "clip_store": self.clip_store,
            "maintenance_root": self.maintenance_root,
            "journal_path": self.journal_path,
            "proof_identity": self.proof_identity.to_dict(),
            "authority_snapshot": self.authority_snapshot.to_dict(),
            "source_state_sha256": self.source_state_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "non_relation_state_sha256": self.non_relation_state_sha256,
            "plan_sha256": self.plan_sha256,
            "relations_before_sha256": self.relations_before_sha256,
            "relations_after_sha256": self.relations_after_sha256,
            "backup_receipt_path": self.backup_receipt_path,
            "backup_receipt_sha256": self.backup_receipt_sha256,
            "delete_event_ids": list(self.delete_event_ids),
            "insert_rows": [list(row) for row in self.insert_rows],
            "quarantine_clip_ids": list(self.quarantine_clip_ids),
            "quarantine": [list(row) for row in self.quarantine],
            "quarantine_sha256": self.quarantine_sha256,
            "deleted_quarantine": list(self.deleted_quarantine),
            "counters": asdict(self.counters),
            "error": self.error,
        }


def journal_sha256(journal: ApplyJournal) -> str:
    encoded = json.dumps(
        journal.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def prepared_journal(
    *,
    authority: RepairAuthority,
    state_db: Path,
    clip_store: Path,
    journal_path: Path,
    binding: OperationBinding,
    quarantine: tuple[tuple[str, str], ...],
    source_state_sha256: str,
    source_identity_sha256: str,
    non_relation_state_sha256: str,
    backup_receipt_path: str,
    backup_receipt_sha256: str,
    plan: RelationPlan,
    counters: RepairCounters,
) -> ApplyJournal:
    return ApplyJournal(
        state="PREPARED",
        state_uid=authority.state_uid,
        state_gid=authority.state_gid,
        state_db_mode=authority.state_db_mode,
        state_dir_mode=authority.state_dir_mode,
        clip_uid=authority.clip_uid,
        clip_gid=authority.clip_gid,
        clip_dir_mode=authority.clip_dir_mode,
        tool_revision=authority.tool_revision,
        authority_sha256=authority.sha256,
        operation_digest_version=binding.version,
        operation_digest=binding.digest,
        quarantine_namespace_sha256=binding.quarantine_namespace_sha256,
        image_artifact_identity=binding.image_artifact_identity,
        state_db=str(state_db.resolve(strict=True)),
        clip_store=str(clip_store.resolve(strict=True)),
        maintenance_root=binding.maintenance_root,
        journal_path=str(journal_path.resolve(strict=False)),
        proof_identity=binding.proof,
        authority_snapshot=binding.snapshot,
        source_state_sha256=source_state_sha256,
        source_identity_sha256=source_identity_sha256,
        non_relation_state_sha256=non_relation_state_sha256,
        plan_sha256=plan.plan_sha256,
        relations_before_sha256=plan.before_sha256,
        relations_after_sha256=plan.after_sha256,
        backup_receipt_path=backup_receipt_path,
        backup_receipt_sha256=backup_receipt_sha256,
        delete_event_ids=plan.delete_event_ids,
        insert_rows=plan.insert_rows,
        quarantine_clip_ids=plan.quarantine_clip_ids,
        quarantine=quarantine,
        quarantine_sha256=quarantine_rows_sha256(quarantine),
        deleted_quarantine=(),
        counters=counters,
    )


def mark_journal(
    journal: ApplyJournal,
    state: JournalState,
    *,
    path: Path,
    maintenance_root: Path,
    authority: RepairAuthority,
    hook: FaultHook | None,
    error: str | None = None,
) -> ApplyJournal:
    if journal.file_identity is not None:
        validate_journal_identity(
            path,
            authority,
            expected_identity=journal.file_identity,
        )
    updated = replace(journal, state=state, error=error, file_identity=None)
    write_journal(
        updated,
        path=path,
        maintenance_root=maintenance_root,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        hook=hook,
        stage=f"journal_{state.lower()}",
    )
    identity = validate_journal_identity(
        path,
        authority,
        expected_sha256=journal_sha256(updated),
    )
    return replace(updated, file_identity=identity)


def read_journal(
    path: Path,
    *,
    maintenance_root: Path,
    authority: RepairAuthority,
    state_db: Path,
    clip_store: Path,
) -> ApplyJournal:
    validate_under_root(path, maintenance_root, allow_missing_leaf=False)
    payload = read_strict_json(
        path,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        exact_mode=0o600,
        error_code="journal_invalid",
    )
    if frozenset(payload) != JOURNAL_KEYS:
        raise ClipConsistencyError("journal_invalid", "journal key set differs")
    journal = ApplyJournal(
        state=journal_state(string(payload, "state")),
        state_uid=integer(payload, "state_uid"),
        state_gid=integer(payload, "state_gid"),
        state_db_mode=integer(payload, "state_db_mode"),
        state_dir_mode=integer(payload, "state_dir_mode"),
        clip_uid=integer(payload, "clip_uid"),
        clip_gid=integer(payload, "clip_gid"),
        clip_dir_mode=integer(payload, "clip_dir_mode"),
        tool_revision=string(payload, "tool_revision"),
        authority_sha256=string(payload, "authority_sha256"),
        operation_digest_version=integer(payload, "operation_digest_version"),
        operation_digest=string(payload, "operation_digest"),
        quarantine_namespace_sha256=string(payload, "quarantine_namespace_sha256"),
        image_artifact_identity=string(payload, "image_artifact_identity"),
        state_db=string(payload, "state_db"),
        clip_store=string(payload, "clip_store"),
        maintenance_root=string(payload, "maintenance_root"),
        journal_path=string(payload, "journal_path"),
        proof_identity=ProofIdentity.from_dict(payload.get("proof_identity")),
        authority_snapshot=AuthoritySnapshot.from_dict(payload.get("authority_snapshot")),
        source_state_sha256=string(payload, "source_state_sha256"),
        source_identity_sha256=string(payload, "source_identity_sha256"),
        non_relation_state_sha256=string(payload, "non_relation_state_sha256"),
        plan_sha256=string(payload, "plan_sha256"),
        relations_before_sha256=string(payload, "relations_before_sha256"),
        relations_after_sha256=string(payload, "relations_after_sha256"),
        backup_receipt_path=string(payload, "backup_receipt_path"),
        backup_receipt_sha256=string(payload, "backup_receipt_sha256"),
        delete_event_ids=strings(payload.get("delete_event_ids")),
        insert_rows=insert_rows(payload.get("insert_rows")),
        quarantine_clip_ids=strings(payload.get("quarantine_clip_ids")),
        quarantine=quarantine(payload.get("quarantine")),
        quarantine_sha256=string(payload, "quarantine_sha256"),
        deleted_quarantine=strings(payload.get("deleted_quarantine")),
        counters=counters(payload.get("counters")),
        error=optional_string(payload, "error"),
    )
    expected_quarantine = canonical_quarantine_rows(
        journal.quarantine_clip_ids,
        journal.quarantine_namespace_sha256,
    )
    if (
        journal.quarantine != expected_quarantine
        or journal.quarantine_sha256 != quarantine_rows_sha256(expected_quarantine)
        or journal.deleted_quarantine
        != tuple(held for _, held in expected_quarantine if held in journal.deleted_quarantine)
    ):
        raise ClipConsistencyError("journal_invalid", "quarantine authority differs")
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
        and _journal_authority(journal) == authority
        and journal.authority_sha256 == authority.sha256
        and journal.state_db == str(state_db.resolve(strict=True))
        and journal.clip_store == str(clip_store.resolve(strict=True))
        and journal.maintenance_root == str(maintenance_root.resolve(strict=True))
        and journal.journal_path == str(path.resolve(strict=True))
        and journal.relation_plan().plan_sha256 == journal.plan_sha256
        and journal.operation_digest_version == 1
        and _is_sha256(journal.operation_digest)
        and _is_sha256(journal.quarantine_namespace_sha256)
        and _is_sha256(journal.image_artifact_identity)
    )
    if not valid_identity:
        raise ClipConsistencyError("journal_invalid", "journal identity differs")
    identity = validate_journal_identity(
        path,
        authority,
        expected_sha256=journal_sha256(journal),
    )
    return replace(journal, file_identity=identity)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _journal_authority(journal: ApplyJournal) -> RepairAuthority:
    return RepairAuthority(
        state_uid=journal.state_uid,
        state_gid=journal.state_gid,
        state_db_mode=journal.state_db_mode,
        state_dir_mode=journal.state_dir_mode,
        clip_uid=journal.clip_uid,
        clip_gid=journal.clip_gid,
        clip_dir_mode=journal.clip_dir_mode,
        tool_revision=journal.tool_revision,
    )


__all__ = [
    "ApplyJournal",
    "journal_sha256",
    "mark_journal",
    "prepared_journal",
    "read_journal",
    "write_journal",
]
