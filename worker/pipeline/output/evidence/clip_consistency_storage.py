"""SQLite mutation and staging quarantine operations for clip consistency repair."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from dataclasses import replace
from pathlib import Path, PurePosixPath

from worker.pipeline.output.evidence.clip_consistency_backup import (
    verify_backup_receipt_for_resume,
)
from worker.pipeline.output.evidence.clip_consistency_database import (
    RelationPlan,
    relation_state_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_io import (
    FaultHook,
    checkpoint,
    sha256_regular,
)
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    journal_sha256,
    mark_journal,
)
from worker.pipeline.output.evidence.clip_consistency_operation import (
    verify_operation_binding,
)
from worker.pipeline.output.evidence.clip_consistency_phase_authority import (
    FileIdentity,
    capture_proof_identity,
    validate_journal_identity,
    validate_phase_authority,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.durability import fsync_directory


def execute_plan(connection: sqlite3.Connection, plan: RelationPlan) -> None:
    if relation_state_sha256(connection) != plan.before_sha256:
        raise ClipConsistencyError("source_changed", "relation state changed before mutation")
    deleted = sum(
        connection.execute(
            "DELETE FROM clip_events WHERE edge_event_id = ?", (event_id,)
        ).rowcount
        for event_id in plan.delete_event_ids
    )
    connection.executemany(
        "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
        plan.insert_rows,
    )
    if deleted != len(plan.delete_event_ids):
        raise ClipConsistencyError("counter_mismatch", "SQL delete count differs from plan")
    if relation_state_sha256(connection) != plan.after_sha256:
        raise ClipConsistencyError("counter_mismatch", "SQL result differs from plan")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "repair introduced FK drift")


def plan_staging_quarantine(
    clip_store: Path,
    quarantine: tuple[tuple[str, str], ...],
) -> list[tuple[Path, Path]]:
    rows = [(clip_store / original, clip_store / held) for original, held in quarantine]
    for _, held in rows:
        _require_available_quarantine(held)
    return rows


def quarantine_staging(
    rows: list[tuple[Path, Path]],
    hook: FaultHook | None,
) -> None:
    moved: list[tuple[Path, Path]] = []
    try:
        for original, held in rows:
            os.replace(original, held)
            moved.append((original, held))
            checkpoint(hook, "quarantine:rename")
        if moved:
            fsync_directory(moved[0][0].parent)
            checkpoint(hook, "quarantine:rename_fsync")
    except BaseException:
        restore_quarantine(moved)
        raise


def restore_quarantine(moved: list[tuple[Path, Path]]) -> None:
    for original, held in reversed(moved):
        if held.exists() and not original.exists():
            os.replace(held, original)
    if moved:
        fsync_directory(moved[0][0].parent)


def ensure_quarantine(clip_store: Path, rows: tuple[tuple[str, str], ...]) -> None:
    moved: list[tuple[Path, Path]] = []
    try:
        for original_relative, held_relative in rows:
            original = clip_store / original_relative
            held = clip_store / held_relative
            if held.exists() and not original.exists():
                continue
            if original.exists() and not held.exists():
                os.replace(original, held)
                moved.append((original, held))
                continue
            _raise_quarantine_state(original, held)
        if rows:
            fsync_directory(clip_store / "clips" / ".staging")
    except BaseException:
        restore_quarantine(moved)
        raise


def open_write_exclusion(state_db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{state_db}?mode=rw",
        uri=True,
        isolation_level=None,
        timeout=0,
    )
    try:
        connection.execute("PRAGMA busy_timeout = 0")
        connection.execute("PRAGMA foreign_keys = ON")
        # Normalize quiescent WAL bytes before taking the raw source identity.
        # A pinned reader may leave frames in WAL; BEGIN IMMEDIATE still excludes
        # every writer and the receipt then binds that exact WAL-aware state.
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        connection.execute("BEGIN IMMEDIATE")
    except BaseException:
        connection.close()
        raise
    return connection


def finish_cleanup(request: RepairRequest, journal: ApplyJournal) -> ApplyJournal:
    assert request.authority is not None
    assert request.quiescence_receipt is not None
    root, path = required_mutation_paths(request)
    for _, held_relative in journal.quarantine:
        if held_relative in journal.deleted_quarantine:
            continue
        checkpoint(request.fault_hook, "quarantine:before_remove")
        _validate_cleanup_phase(request, journal)
        validate_journal_identity(
            path,
            request.authority,
            expected_sha256=journal_sha256(journal),
            expected_identity=journal.file_identity,
        )
        checkpoint(request.fault_hook, "quarantine:before_descriptor_remove")
        _remove_quarantine_descriptor_backed(
            request.clip_store,
            held_relative,
            _expected_quarantine_identities(journal, held_relative),
        )
        fsync_directory(request.clip_store / "clips" / ".staging")
        journal = replace(
            journal,
            deleted_quarantine=(*journal.deleted_quarantine, held_relative),
        )
        journal = mark_journal(
            journal,
            "DB_COMMITTED",
            path=path,
            maintenance_root=root,
            authority=request.authority,
            hook=request.fault_hook,
        )
        checkpoint(request.fault_hook, "quarantine:after_remove")
    if journal.quarantine:
        checkpoint(request.fault_hook, "quarantine:fsync_directory")
    _validate_cleanup_phase(request, journal)
    validate_journal_identity(
        path,
        request.authority,
        expected_sha256=journal_sha256(journal),
        expected_identity=journal.file_identity,
    )
    done = replace(
        journal,
        counters=replace(
            journal.counters,
            staging_after=0,
            staging_to_delete=0,
            staging_deleted=journal.counters.staging_before,
        ),
    )
    done = mark_journal(
        done,
        "DONE",
        path=path,
        maintenance_root=root,
        authority=request.authority,
        hook=request.fault_hook,
    )
    try:
        validate_journal_identity(
            path,
            request.authority,
            expected_sha256=journal_sha256(done),
            expected_identity=done.file_identity,
        )
        _validate_cleanup_phase(request, done)
    except ClipConsistencyError:
        # A safe journal can be downgraded to DB_COMMITTED so no DONE state is
        # exposed when another bound authority drifts during the DONE write.
        validate_journal_identity(
            path,
            request.authority,
            expected_sha256=journal_sha256(done),
            expected_identity=done.file_identity,
        )
        mark_journal(
            done,
            "DB_COMMITTED",
            path=path,
            maintenance_root=root,
            authority=request.authority,
            hook=None,
        )
        raise
    return done


def required_mutation_paths(request: RepairRequest) -> tuple[Path, Path]:
    assert request.maintenance_root is not None
    assert request.journal_path is not None
    return request.maintenance_root, request.journal_path


def _validate_cleanup_phase(request: RepairRequest, journal: ApplyJournal) -> None:
    assert request.authority is not None
    assert request.quiescence_receipt is not None
    root, _ = required_mutation_paths(request)
    proof = capture_proof_identity(request.quiescence_receipt, request.authority)
    if proof != journal.proof_identity:
        raise ClipConsistencyError("authority_drift", "quiescence proof identity changed")
    validate_phase_authority(
        journal.authority_snapshot,
        state_db=request.state_db,
        clip_store=request.clip_store,
        maintenance_root=root,
        tracked_maintenance=tuple(
            Path(identity.path)
            for identity in journal.authority_snapshot.maintenance
            if identity.path != journal.maintenance_root
        ),
        authority=request.authority,
        quarantine=journal.quarantine,
        quarantine_state="held",
        deleted_quarantine=journal.deleted_quarantine,
    )
    receipt_path = Path(journal.backup_receipt_path)
    if sha256_regular(receipt_path) != journal.backup_receipt_sha256:
        raise ClipConsistencyError("backup_receipt_invalid", "journal receipt hash differs")
    backup = verify_backup_receipt_for_resume(
        receipt_path,
        maintenance_root=root,
        clip_store=request.clip_store,
        authority=request.authority,
    )
    verify_operation_binding(
        version=journal.operation_digest_version,
        digest=journal.operation_digest,
        quarantine_namespace_sha256=journal.quarantine_namespace_sha256,
        image_identity=journal.image_artifact_identity,
        snapshot=journal.authority_snapshot,
        proof=journal.proof_identity,
        maintenance_root_path=journal.maintenance_root,
        authority=request.authority,
        state_db=request.state_db,
        clip_store=request.clip_store,
        maintenance_root=root,
        journal_path=Path(journal.journal_path),
        backup=backup,
        non_relation_state_sha256=journal.non_relation_state_sha256,
        plan=journal.relation_plan(),
        quarantine=journal.quarantine,
    )


def _remove_quarantine_descriptor_backed(
    clip_store: Path,
    relative: str,
    expected: dict[str, FileIdentity],
) -> None:
    path = Path(relative)
    parent = clip_store / path.parent
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        child = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            _require_expected_descriptor(child, expected[relative])
            _remove_directory_contents(child, PurePosixPath(relative), expected)
            observed = os.fstat(child)
            current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) != (current.st_dev, current.st_ino):
                raise ClipConsistencyError(
                    "authority_drift", "quarantine identity changed before deletion"
                )
        finally:
            os.close(child)
        os.rmdir(path.name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ClipConsistencyError("authority_drift", "quarantine deletion refused") from exc
    finally:
        os.close(parent_descriptor)


def _remove_directory_contents(
    descriptor: int,
    relative: PurePosixPath,
    expected: dict[str, FileIdentity],
) -> None:
    names = sorted(os.listdir(descriptor))
    expected_names = sorted(
        path.parts[len(relative.parts)]
        for logical in expected
        if len((path := PurePosixPath(logical)).parts) == len(relative.parts) + 1
        and path.parts[: len(relative.parts)] == relative.parts
    )
    if names != expected_names:
        raise ClipConsistencyError("authority_drift", "quarantine entries changed")
    for name in names:
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ClipConsistencyError("authority_drift", "quarantine contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                child_relative = relative / name
                _require_expected_descriptor(child, expected[child_relative.as_posix()])
                _remove_directory_contents(child, child_relative, expected)
                _require_open_name_identity(descriptor, name, opened)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(info.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                _require_expected_descriptor(
                    child, expected[(relative / name).as_posix()]
                )
                _require_open_name_identity(descriptor, name, opened)
            finally:
                os.close(child)
            os.unlink(name, dir_fd=descriptor)
        else:
            raise ClipConsistencyError("authority_drift", "quarantine entry type changed")


def _expected_quarantine_identities(
    journal: ApplyJournal, held: str
) -> dict[str, FileIdentity]:
    original = next(
        original for original, candidate in journal.quarantine if candidate == held
    )
    result: dict[str, FileIdentity] = {}
    for identity in journal.authority_snapshot.clip_entries:
        if identity.path == original or identity.path.startswith(f"{original}/"):
            logical = f"{held}{identity.path[len(original):]}"
            result[logical] = replace(identity, path=logical)
    if held not in result:
        raise ClipConsistencyError("authority_drift", "quarantine identity is missing")
    return result


def _require_expected_descriptor(descriptor: int, expected: FileIdentity) -> None:
    info = os.fstat(descriptor)
    observed = (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_uid,
        info.st_gid,
        stat.S_IMODE(info.st_mode),
    )
    required = (
        expected.device,
        expected.inode,
        expected.file_type,
        expected.uid,
        expected.gid,
        expected.mode,
    )
    if observed != required:
        raise ClipConsistencyError("authority_drift", "quarantine identity changed")
    if expected.content_sha256 is not None:
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected.content_sha256:
            raise ClipConsistencyError("authority_drift", "quarantine content changed")


def _require_open_name_identity(
    parent_descriptor: int, name: str, opened: os.stat_result
) -> None:
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        opened.st_dev,
        opened.st_ino,
        stat.S_IFMT(opened.st_mode),
        opened.st_uid,
        opened.st_gid,
        stat.S_IMODE(opened.st_mode),
    ) != (
        current.st_dev,
        current.st_ino,
        stat.S_IFMT(current.st_mode),
        current.st_uid,
        current.st_gid,
        stat.S_IMODE(current.st_mode),
    ):
        raise ClipConsistencyError(
            "authority_drift", "quarantine entry changed during deletion"
        )


def _require_available_quarantine(held: Path) -> None:
    if held.exists():
        raise ClipConsistencyError("quarantine_conflict", "quarantine already exists")


def _raise_quarantine_state(original: Path, held: Path) -> None:
    if not original.exists() and not held.exists():
        raise ClipConsistencyError("quarantine_missing", "prepared staging is missing")
    raise ClipConsistencyError("quarantine_conflict", "both staging paths exist")


__all__ = [
    "ensure_quarantine",
    "execute_plan",
    "finish_cleanup",
    "open_write_exclusion",
    "plan_staging_quarantine",
    "quarantine_staging",
    "required_mutation_paths",
    "restore_quarantine",
]
