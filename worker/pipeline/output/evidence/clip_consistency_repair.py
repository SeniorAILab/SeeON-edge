"""Authoritative-final-manifest repair for schema-9 clip relations."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

from worker.pipeline.output.evidence.clip_consistency_backup import ensure_prebackup
from worker.pipeline.output.evidence.clip_consistency_database import (
    RelationPlan,
    plan_relations,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairCounters,
    RepairReceipt,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock
from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest,
    verify_ready_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

_RECEIPT_VERSION: Final = 1


def repair_clip_consistency(request: RepairRequest) -> RepairReceipt:
    store = request.store_dir
    _validate_directory(store, "store root")
    try:
        with ClipStoreLock.acquire(store):
            return _repair_locked(request, store)
    except sqlite3.Error as exc:
        raise ClipConsistencyError("database_error", "SQLite operation failed") from exc


def _repair_locked(request: RepairRequest, store: Path) -> RepairReceipt:
    database = store / "worker-state.sqlite3"
    _validate_regular(database, "database")
    connection = sqlite3.connect(
        f"file:{database}?mode=rw", uri=True, isolation_level=None
    )
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        validate_database(connection, now=time.time())
        desired, ready_count, unavailable_count, staging = _scan_finals(
            store, ffprobe_bin=request.ffprobe_bin
        )
        plan = plan_relations(connection, desired)
        counters = RepairCounters(
            ready_finals=ready_count,
            unavailable_finals=unavailable_count,
            relations_before=plan.relations_before,
            relations_after=plan.relations_after,
            relations_deleted=len(plan.delete_event_ids),
            relations_inserted=len(plan.insert_rows),
            staging_before=len(staging),
            staging_after=len(staging),
            staging_to_delete=len(staging),
            staging_deleted=0,
        )
        if not request.apply:
            return RepairReceipt(
                _RECEIPT_VERSION, "dry-run", SCHEMA_VERSION, counters, None, None
            )
        backup = ensure_prebackup(
            database,
            connection,
            receipt_path=request.prebackup_receipt,
            backup_dir=request.backup_dir or store / "maintenance-backups",
        )
        validate_database(connection, now=time.time())
        _apply(connection, plan, staging)
        applied = replace(
            counters,
            staging_after=0,
            staging_deleted=len(staging),
            staging_to_delete=0,
        )
        receipt_path = _new_receipt_path(request, store)
        receipt = RepairReceipt(
            _RECEIPT_VERSION,
            "apply",
            SCHEMA_VERSION,
            applied,
            backup.receipt_path,
            str(receipt_path.resolve()),
        )
        _write_receipt(receipt_path, receipt)
        return receipt
    finally:
        connection.close()


def _scan_finals(
    store: Path, *, ffprobe_bin: str
) -> tuple[dict[str, tuple[str, ...]], int, int, tuple[Path, ...]]:
    clips_root = store / "clips"
    staging_root = clips_root / ".staging"
    _validate_directory(clips_root, "clips root")
    _validate_directory(staging_root, "staging root")
    for entry in staging_root.iterdir():
        if entry.is_symlink():
            raise ClipConsistencyError("unsafe_path", "staging contains a symlink")
    desired: dict[str, tuple[str, ...]] = {}
    ready = unavailable = 0
    for clip_dir in sorted(clips_root.iterdir(), key=lambda candidate: candidate.name):
        if clip_dir.name == ".staging":
            continue
        _validate_directory(clip_dir, "final clip")
        if clip_dir.parent.resolve() != clips_root.resolve():
            raise ClipConsistencyError("unsafe_path", "final clip escaped clips root")
        for child in clip_dir.iterdir():
            if child.is_symlink():
                raise ClipConsistencyError("unsafe_path", "final clip contains a symlink")
        try:
            manifest = parse_manifest(clip_dir / "manifest.json")
            if manifest.clip_id != clip_dir.name:
                raise ClipConsistencyError("final_invalid", "final identity mismatch")
            match manifest:
                case ReadyClipManifest():
                    media = clip_dir / "clip.mp4"
                    _validate_regular(media, "final media")
                    verify_ready_manifest(manifest, media, ffprobe_bin=ffprobe_bin)
                    ready += 1
                case UnavailableClipManifest():
                    if (clip_dir / "clip.mp4").exists():
                        raise ClipConsistencyError(
                            "final_invalid", "unavailable final contains media"
                        )
                    unavailable += 1
            desired[clip_dir.name] = tuple(manifest.event_refs)
        except ClipEvidenceError as exc:
            raise ClipConsistencyError("final_invalid", "final manifest or media invalid") from exc
    overlaps: list[Path] = []
    for clip_id in desired:
        candidate = staging_root / clip_id
        if candidate.exists() or candidate.is_symlink():
            _validate_directory(candidate, "same-ID staging")
            overlaps.append(candidate)
    return desired, ready, unavailable, tuple(overlaps)


def _apply(
    connection: sqlite3.Connection,
    plan: RelationPlan,
    staging: tuple[Path, ...],
) -> None:
    moved: list[tuple[Path, Path]] = []
    token = f"{os.getpid()}-{time.time_ns()}"
    try:
        for original in staging:
            quarantine = original.parent / f".clip-consistency-{token}-{original.name}"
            os.replace(original, quarantine)
            moved.append((original, quarantine))
        if moved:
            fsync_directory(staging[0].parent)
        connection.execute("BEGIN IMMEDIATE")
        deleted = 0
        for event_id in plan.delete_event_ids:
            deleted += connection.execute(
                "DELETE FROM clip_events WHERE edge_event_id = ?", (event_id,)
            ).rowcount
        connection.executemany(
            "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
            plan.insert_rows,
        )
        after = int(
            connection.execute("SELECT COUNT(*) FROM clip_events").fetchone()[0]
        )
        _verify_applied(connection, plan, deleted, after)
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        for original, quarantine in reversed(moved):
            if quarantine.exists():
                os.replace(quarantine, original)
        if moved:
            fsync_directory(staging[0].parent)
        raise
    for _, quarantine in moved:
        shutil.rmtree(quarantine)
    if moved:
        fsync_directory(staging[0].parent)


def _verify_applied(
    connection: sqlite3.Connection,
    plan: RelationPlan,
    deleted: int,
    after: int,
) -> None:
    if deleted != len(plan.delete_event_ids) or after != plan.relations_after:
        raise ClipConsistencyError("counter_mismatch", "relation counters changed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "repair introduced FK drift")


def _validate_directory(path: Path, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ClipConsistencyError("unsafe_path", f"{label} unavailable") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ClipConsistencyError("unsafe_path", f"{label} is unsafe")


def _validate_regular(path: Path, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ClipConsistencyError("unsafe_path", f"{label} unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ClipConsistencyError("unsafe_path", f"{label} is unsafe")


def _new_receipt_path(request: RepairRequest, store: Path) -> Path:
    directory = request.receipt_dir or store / "maintenance-receipts"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _validate_directory(directory, "receipt directory")
    return directory / f"clip-consistency-{time.time_ns()}.receipt.json"


def _write_receipt(path: Path, receipt: RepairReceipt) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(receipt.to_dict(), output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


__all__ = ["repair_clip_consistency"]
