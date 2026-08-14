"""Apply and resume state machine for clip consistency repair."""

from __future__ import annotations

import time
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_backup import (
    ensure_prebackup,
    verify_backup_receipt_for_resume,
)
from worker.pipeline.output.evidence.clip_consistency_database import (
    plan_relations,
    relation_state_sha256,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_io import checkpoint, sha256_regular
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    mark_journal,
    prepared_journal,
    read_journal,
    write_journal,
)
from worker.pipeline.output.evidence.clip_consistency_manifest import (
    ManifestAuthority,
    scan_manifest_authority,
)
from worker.pipeline.output.evidence.clip_consistency_receipt import (
    build_counters,
    journal_receipt,
)
from worker.pipeline.output.evidence.clip_consistency_storage import (
    ensure_quarantine,
    execute_plan,
    finish_cleanup,
    open_write_exclusion,
    plan_staging_quarantine,
    quarantine_staging,
    required_mutation_paths,
    restore_quarantine,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairReceipt,
    RepairRequest,
)


def apply_repair(request: RepairRequest) -> RepairReceipt:
    root, journal_path = required_mutation_paths(request)
    if journal_path.exists():
        raise ClipConsistencyError("journal_exists", "use --resume for existing journal")
    control = open_write_exclusion(request.state_db)
    moved: list[tuple[Path, Path]] = []
    journal: ApplyJournal | None = None
    committed = False
    try:
        validate_database(control, now=time.time())
        authority = _authority(request)
        plan = plan_relations(control, authority.desired)
        counters = build_counters(plan, authority)
        backup = ensure_prebackup(
            request.state_db,
            control,
            receipt_path=request.prebackup_receipt,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
        )
        moved = plan_staging_quarantine(authority.staging, plan.plan_sha256)
        journal = prepared_journal(
            owner_uid=request.expected_owner_uid,
            state_db=request.state_db,
            clip_store=request.clip_store,
            journal_path=journal_path,
            source_state_sha256=backup.source_state_sha256,
            backup_receipt_path=backup.receipt_path,
            backup_receipt_sha256=sha256_regular(Path(backup.receipt_path)),
            plan=plan,
            quarantine=_relative_quarantine(request.clip_store, moved),
            counters=counters,
        )
        write_journal(
            journal,
            path=journal_path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
            stage="journal_prepared",
        )
        quarantine_staging(moved, request.fault_hook)
        checkpoint(request.fault_hook, "apply:before_relations")
        execute_plan(control, plan)
        checkpoint(request.fault_hook, "apply:before_commit")
        control.commit()
        committed = True
        checkpoint(request.fault_hook, "apply:after_commit")
        journal = mark_journal(
            journal,
            "DB_COMMITTED",
            path=journal_path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
        )
        return journal_receipt("apply", finish_cleanup(request, journal))
    except BaseException as exc:
        if not committed:
            _abort_precommit(request, control, moved, journal, exc)
        raise
    finally:
        control.close()


def resume_repair(request: RepairRequest) -> RepairReceipt:
    root, journal_path = required_mutation_paths(request)
    journal = read_journal(
        journal_path,
        maintenance_root=root,
        expected_uid=request.expected_owner_uid,
        state_db=request.state_db,
        clip_store=request.clip_store,
    )
    _verify_journal_backup(journal, root, request.expected_owner_uid)
    if journal.state in {"DONE", "ABORTED"}:
        return journal_receipt("resume", journal)
    control = open_write_exclusion(request.state_db)
    try:
        validate_database(control, now=time.time())
        current = relation_state_sha256(control)
        if journal.state == "PREPARED":
            journal = _resume_prepared(request, journal, control, current)
        else:
            if current != journal.relations_after_sha256:
                raise ClipConsistencyError(
                    "resume_conflict", "committed relation state differs"
                )
            control.rollback()
    finally:
        control.close()
    return journal_receipt("resume", finish_cleanup(request, journal))


def _resume_prepared(
    request: RepairRequest,
    journal: ApplyJournal,
    control: object,
    current: str,
) -> ApplyJournal:
    # The caller supplies an open sqlite3.Connection; object avoids an import-only dependency.
    import sqlite3

    assert isinstance(control, sqlite3.Connection)
    if current == journal.relations_before_sha256:
        ensure_quarantine(request.clip_store, journal.quarantine)
        execute_plan(control, journal.relation_plan())
        control.commit()
        checkpoint(request.fault_hook, "resume:after_commit")
    elif current == journal.relations_after_sha256:
        control.rollback()
    else:
        raise ClipConsistencyError("resume_conflict", "relation state is neither boundary")
    root, path = required_mutation_paths(request)
    return mark_journal(
        journal,
        "DB_COMMITTED",
        path=path,
        maintenance_root=root,
        expected_uid=request.expected_owner_uid,
        hook=request.fault_hook,
    )


def _abort_precommit(
    request: RepairRequest,
    control: object,
    moved: list[tuple[Path, Path]],
    journal: ApplyJournal | None,
    error: BaseException,
) -> None:
    import sqlite3

    assert isinstance(control, sqlite3.Connection)
    if control.in_transaction:
        control.rollback()
    restore_quarantine(moved)
    root, path = required_mutation_paths(request)
    if journal is not None and path.exists():
        mark_journal(
            journal,
            "ABORTED",
            path=path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=None,
            error=type(error).__name__,
        )


def _verify_journal_backup(journal: ApplyJournal, root: Path, uid: int) -> None:
    receipt_path = Path(journal.backup_receipt_path)
    if sha256_regular(receipt_path) != journal.backup_receipt_sha256:
        raise ClipConsistencyError("backup_receipt_invalid", "journal receipt hash differs")
    backup = verify_backup_receipt_for_resume(
        receipt_path,
        maintenance_root=root,
        expected_uid=uid,
    )
    if backup.source_state_sha256 != journal.source_state_sha256:
        raise ClipConsistencyError("backup_receipt_invalid", "journal source state differs")


def _relative_quarantine(
    clip_store: Path,
    moved: list[tuple[Path, Path]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            original.relative_to(clip_store).as_posix(),
            held.relative_to(clip_store).as_posix(),
        )
        for original, held in moved
    )


def _authority(request: RepairRequest) -> ManifestAuthority:
    return scan_manifest_authority(
        request.clip_store,
        expected_uid=request.expected_owner_uid,
        ffprobe_bin=request.ffprobe_bin,
    )


__all__ = ["apply_repair", "resume_repair"]
