"""Typed contracts for schema-9 clip consistency maintenance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from worker.pipeline.output.evidence.clip_consistency_authority_types import (
    RepairAuthority,
)

JsonScalar: TypeAlias = str | int | bool | None
JournalState: TypeAlias = Literal[
    "PREPARED", "DB_COMMITTED", "DONE", "ABORTED", "UNKNOWN"
]
FaultHook: TypeAlias = Callable[[str], None]


@dataclass(slots=True)
class ClipConsistencyError(Exception):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class RepairCounters:
    ready_finals: int
    unavailable_finals: int
    relations_before: int
    relations_after: int
    mismatch_clips: int
    mismatch_tuples: int
    sql_relations_deleted: int
    sql_relations_inserted: int
    staging_before: int
    staging_after: int
    staging_to_delete: int
    staging_deleted: int

    @property
    def changes(self) -> int:
        staging = max(self.staging_to_delete, self.staging_deleted)
        return self.mismatch_tuples + staging


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    format_version: int
    schema_version: int
    state_uid: int
    state_gid: int
    state_db_mode: int
    state_dir_mode: int
    clip_uid: int
    clip_gid: int
    clip_dir_mode: int
    tool_revision: str
    authority_sha256: str
    clip_store: str
    source_path: str
    source_mode: int
    source_size: int
    source_file_sha256: str
    source_wal_path: str
    source_wal_present: bool
    source_wal_size: int
    source_wal_sha256: str
    source_state_sha256: str
    source_identity_sha256: str
    backup_path: str
    backup_mode: int
    backup_size: int
    backup_file_sha256: str
    backup_state_sha256: str
    receipt_path: str
    operation_digest_version: int
    operation_digest: str

    def to_dict(self) -> dict[str, JsonScalar]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepairReceipt:
    format_version: int
    mode: str
    state: JournalState | Literal["DRY_RUN"]
    schema_version: int
    counters: RepairCounters
    backup_receipt_path: str | None
    journal_path: str | None
    operation_digest_version: int | None = None
    operation_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "mode": self.mode,
            "state": self.state,
            "schema_version": self.schema_version,
            "counters": asdict(self.counters),
            "changes": self.counters.changes,
            "backup_receipt_path": self.backup_receipt_path,
            "journal_path": self.journal_path,
            "operation_digest_version": self.operation_digest_version,
            "operation_digest": self.operation_digest,
        }


@dataclass(frozen=True, slots=True)
class RepairRequest:
    state_db: Path
    clip_store: Path
    authority: RepairAuthority | None = None
    apply: bool = False
    resume: bool = False
    maintenance_root: Path | None = None
    journal_path: Path | None = None
    quiescence_receipt: Path | None = None
    prebackup_receipt: Path | None = None
    ffprobe_bin: str = "ffprobe"
    fault_hook: FaultHook | None = field(default=None, repr=False, compare=False)


__all__ = [
    "BackupReceipt",
    "ClipConsistencyError",
    "FaultHook",
    "JournalState",
    "RepairCounters",
    "RepairReceipt",
    "RepairRequest",
]
