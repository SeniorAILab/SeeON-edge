"""Typed reports for the schema-9 clip consistency maintenance command."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | bool | None


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
    relations_deleted: int
    relations_inserted: int
    staging_before: int
    staging_after: int
    staging_to_delete: int
    staging_deleted: int

    @property
    def changes(self) -> int:
        staging_changes = max(self.staging_to_delete, self.staging_deleted)
        return self.relations_deleted + self.relations_inserted + staging_changes


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    format_version: int
    schema_version: int
    source_sha256: str
    source_mode: int
    backup_sha256: str
    backup_mode: int
    backup_path: str
    receipt_path: str

    def to_dict(self) -> dict[str, JsonScalar]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepairReceipt:
    format_version: int
    mode: str
    schema_version: int
    counters: RepairCounters
    backup_receipt_path: str | None
    receipt_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "counters": asdict(self.counters),
            "changes": self.counters.changes,
            "backup_receipt_path": self.backup_receipt_path,
            "receipt_path": self.receipt_path,
        }


@dataclass(frozen=True, slots=True)
class RepairRequest:
    store_dir: Path
    apply: bool = False
    prebackup_receipt: Path | None = None
    backup_dir: Path | None = None
    receipt_dir: Path | None = None
    ffprobe_bin: str = "ffprobe"


__all__ = [
    "BackupReceipt",
    "ClipConsistencyError",
    "RepairCounters",
    "RepairReceipt",
    "RepairRequest",
]
