"""Verified online SQLite prebackups for clip consistency apply mode."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from worker.pipeline.output.evidence.clip_consistency_types import (
    BackupReceipt,
    ClipConsistencyError,
)
from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

_RECEIPT_VERSION: Final = 1


def ensure_prebackup(
    source: Path,
    connection: sqlite3.Connection,
    *,
    receipt_path: Path | None,
    backup_dir: Path,
) -> BackupReceipt:
    checkpoint = cast(
        "tuple[int, int, int] | None",
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone(),
    )
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise ClipConsistencyError("backup_checkpoint_failed", "WAL checkpoint was busy")
    source_hash = _sha256_regular(source)
    if receipt_path is not None:
        return _verify_receipt(receipt_path, source, source_hash)
    return _create_backup(source, connection, source_hash, backup_dir)


def _create_backup(
    source: Path,
    connection: sqlite3.Connection,
    source_hash: str,
    backup_dir: Path,
) -> BackupReceipt:
    _safe_directory(backup_dir)
    token = f"{time.time_ns()}-{source_hash[:12]}"
    backup_path = backup_dir / f"worker-state-schema9-{token}.sqlite3"
    receipt_path = backup_dir / f"worker-state-schema9-{token}.receipt.json"
    try:
        destination = sqlite3.connect(backup_path)
        try:
            connection.backup(destination)
        finally:
            destination.close()
        os.chmod(backup_path, 0o600)
        fsync_file(backup_path)
        fsync_directory(backup_dir)
        backup_hash = _sha256_regular(backup_path)
        _verify_backup_database(backup_path)
        receipt = BackupReceipt(
            format_version=_RECEIPT_VERSION,
            schema_version=SCHEMA_VERSION,
            source_sha256=source_hash,
            source_mode=stat.S_IMODE(source.stat(follow_symlinks=False).st_mode),
            backup_sha256=backup_hash,
            backup_mode=0o600,
            backup_path=str(backup_path.resolve()),
            receipt_path=str(receipt_path.resolve()),
        )
        _write_new_json(receipt_path, receipt.to_dict())
    except Exception:
        backup_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
    else:
        return receipt


def _verify_receipt(path: Path, source: Path, source_hash: str) -> BackupReceipt:
    receipt_info = _regular_info(path)
    if stat.S_IMODE(receipt_info.st_mode) != 0o600:
        raise ClipConsistencyError("backup_mode_invalid", "receipt is not owner-only")
    payload = _read_json_regular(path)
    receipt = BackupReceipt(
        format_version=_required_int(payload, "format_version"),
        schema_version=_required_int(payload, "schema_version"),
        source_sha256=_required_str(payload, "source_sha256"),
        source_mode=_required_int(payload, "source_mode"),
        backup_sha256=_required_str(payload, "backup_sha256"),
        backup_mode=_required_int(payload, "backup_mode"),
        backup_path=_required_str(payload, "backup_path"),
        receipt_path=str(path.resolve()),
    )
    backup = Path(receipt.backup_path)
    if receipt.format_version != _RECEIPT_VERSION or receipt.schema_version != SCHEMA_VERSION:
        raise ClipConsistencyError("backup_receipt_invalid", "receipt version mismatch")
    if receipt.source_sha256 != source_hash:
        raise ClipConsistencyError("backup_stale", "receipt does not match current database")
    if source.resolve() == backup.resolve():
        raise ClipConsistencyError("backup_receipt_invalid", "backup aliases source database")
    info = _regular_info(backup)
    if stat.S_IMODE(info.st_mode) != 0o600 or receipt.backup_mode != 0o600:
        raise ClipConsistencyError("backup_mode_invalid", "backup is not owner-only")
    if _sha256_regular(backup) != receipt.backup_sha256:
        raise ClipConsistencyError("backup_hash_mismatch", "backup hash mismatch")
    _verify_backup_database(backup)
    return receipt


def _verify_backup_database(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        version_row = cast(
            "tuple[int] | None", connection.execute("PRAGMA user_version").fetchone()
        )
        version = -1 if version_row is None else version_row[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    if version != SCHEMA_VERSION or integrity != [("ok",)] or foreign_keys:
        raise ClipConsistencyError("backup_verification_failed", "backup database check failed")


def _safe_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ClipConsistencyError("unsafe_path", "backup directory is unsafe")


def _regular_info(path: Path) -> os.stat_result:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ClipConsistencyError("unsafe_path", "required file is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ClipConsistencyError("unsafe_path", "required file is not regular")
    return info


def _sha256_regular(path: Path) -> str:
    _regular_info(path)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_json_regular(path: Path) -> dict[str, object]:
    _regular_info(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            raw = b""
            while chunk := os.read(descriptor, 8192):
                raw += chunk
                if len(raw) > 64 * 1024:
                    raise OSError("receipt too large")
        finally:
            os.close(descriptor)
        loaded: object = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClipConsistencyError("backup_receipt_invalid", "receipt cannot be read") from exc
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) for key in loaded
    ):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt is not an object")
    return cast("dict[str, object]", loaded)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt fields invalid")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt fields invalid")
    return value


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


__all__ = ["ensure_prebackup"]
