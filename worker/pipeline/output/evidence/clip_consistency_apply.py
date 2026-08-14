"""Apply and resume state machine for clip consistency repair."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_backup import (
    bind_backup_receipt,
    ensure_prebackup,
    verify_backup_receipt_for_resume,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    abort_precommit as _abort_precommit,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    classify_ambiguous_commit as _classify_ambiguous_commit,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    commit_connection as _commit_connection,
)
from worker.pipeline.output.evidence.clip_consistency_database import (
    plan_relations,
    relation_state_sha256,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_io import checkpoint, sha256_regular
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    journal_sha256,
    mark_journal,
    prepared_journal,
    read_journal,
    write_journal,
)
from worker.pipeline.output.evidence.clip_consistency_manifest import (
    ManifestAuthority,
    scan_manifest_authority,
)
from worker.pipeline.output.evidence.clip_consistency_operation import (
    build_operation_binding,
    verify_operation_binding,
)
from worker.pipeline.output.evidence.clip_consistency_phase_authority import (
    ProofIdentity,
    capture_authority_snapshot,
    capture_proof_identity,
    validate_journal_identity,
    validate_phase_authority,
)
from worker.pipeline.output.evidence.clip_consistency_preimage import (
    non_relation_preimage_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_quiescence import (
    bind_quiescence_receipt,
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
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    BackupReceipt,
    ClipConsistencyError,
    RepairReceipt,
    RepairRequest,
)


def apply_repair(request: RepairRequest, proof: ProofIdentity) -> RepairReceipt:
    assert request.authority is not None
    assert request.quiescence_receipt is not None
    root, journal_path = required_mutation_paths(request)
    if journal_path.exists():
        raise ClipConsistencyError("journal_exists", "use --resume for existing journal")
    control: sqlite3.Connection | None = open_write_exclusion(request.state_db)
    moved: list[tuple[Path, Path]] = []
    journal: ApplyJournal | None = None
    committed = False
    commit_attempted = False
    try:
        initial_snapshot = capture_authority_snapshot(
            state_db=request.state_db,
            clip_store=request.clip_store,
            maintenance_root=root,
            tracked_maintenance=(request.quiescence_receipt,),
            authority=request.authority,
        )
        validate_database(control, now=time.time())
        authority = _authority(request)
        plan = plan_relations(
            control,
            authority.desired,
            quarantine_clip_ids=tuple(path.name for path in authority.staging),
        )
        counters = build_counters(plan, authority)
        backup = ensure_prebackup(
            request.state_db,
            control,
            receipt_path=request.prebackup_receipt,
            maintenance_root=root,
            clip_store=request.clip_store,
            authority=request.authority,
            hook=request.fault_hook,
        )
        validate_phase_authority(
            initial_snapshot,
            state_db=request.state_db,
            clip_store=request.clip_store,
            maintenance_root=root,
            tracked_maintenance=(request.quiescence_receipt,),
            authority=request.authority,
            quarantine=(),
            quarantine_state="original",
        )
        _require_unchanged_manifest_authority(_authority(request), authority)
        snapshot = capture_authority_snapshot(
            state_db=request.state_db,
            clip_store=request.clip_store,
            maintenance_root=root,
            tracked_maintenance=(
                request.quiescence_receipt,
                Path(backup.backup_path),
            ),
            authority=request.authority,
        )
        non_relation_sha256 = non_relation_preimage_sha256(control)
        binding, quarantine = build_operation_binding(
            authority=request.authority,
            state_db=request.state_db,
            clip_store=request.clip_store,
            maintenance_root=root,
            journal_path=journal_path,
            proof=proof,
            backup=backup,
            snapshot=snapshot,
            non_relation_state_sha256=non_relation_sha256,
            plan=plan,
        )
        bound_proof = bind_quiescence_receipt(
            request.quiescence_receipt,
            binding.digest,
            authority=request.authority,
            expected_identity=proof,
        )
        _require_bound_proof(bound_proof, binding.proof)
        backup = bind_backup_receipt(
            backup,
            binding.digest,
            maintenance_root=root,
            authority=request.authority,
            hook=request.fault_hook,
            expected_identity=binding.backup_receipt_file_identity,
        )
        moved = plan_staging_quarantine(request.clip_store, quarantine)
        journal = prepared_journal(
            authority=request.authority,
            state_db=request.state_db,
            clip_store=request.clip_store,
            journal_path=journal_path,
            binding=binding,
            quarantine=quarantine,
            source_state_sha256=backup.source_state_sha256,
            source_identity_sha256=backup.source_identity_sha256,
            non_relation_state_sha256=non_relation_sha256,
            backup_receipt_path=backup.receipt_path,
            backup_receipt_sha256=sha256_regular(Path(backup.receipt_path)),
            plan=plan,
            counters=counters,
        )
        verified_backup = _verify_journal_backup(journal, root, request)
        _verify_operation(journal, verified_backup, request)
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
            quarantine_state="original",
        )
        write_journal(
            journal,
            path=journal_path,
            maintenance_root=root,
            expected_uid=request.authority.state_uid,
            expected_gid=request.authority.state_gid,
            hook=request.fault_hook,
            stage="journal_prepared",
        )
        journal = replace(
            journal,
            file_identity=validate_journal_identity(
                journal_path,
                request.authority,
                expected_sha256=journal_sha256(journal),
            ),
        )
        _validate_phase(request, journal, "original")
        quarantine_staging(moved, request.fault_hook)
        checkpoint(request.fault_hook, "apply:before_relations")
        execute_plan(control, plan)
        checkpoint(request.fault_hook, "apply:before_commit")
        _validate_phase(request, journal, "held")
        validate_journal_identity(
            journal_path,
            request.authority,
            expected_sha256=journal_sha256(journal),
            expected_identity=journal.file_identity,
        )
        commit_attempted = True
        try:
            _commit_connection(control)
        except BaseException as commit_error:
            control.close()
            control = None
            _classify_ambiguous_commit(request, journal, moved, commit_error)
            raise
        committed = True
        checkpoint(request.fault_hook, "apply:after_commit")
        _validate_phase(request, journal, "held")
        validate_journal_identity(
            journal_path,
            request.authority,
            expected_sha256=journal_sha256(journal),
            expected_identity=journal.file_identity,
        )
        journal = mark_journal(
            journal,
            "DB_COMMITTED",
            path=journal_path,
            maintenance_root=root,
            authority=request.authority,
            hook=request.fault_hook,
        )
        return journal_receipt("apply", finish_cleanup(request, journal))
    except BaseException as exc:
        if not committed and not commit_attempted:
            assert control is not None
            if _is_security_drift(exc):
                if control.in_transaction:
                    control.rollback()
            else:
                _abort_precommit(request, control, moved, journal, exc)
        raise
    finally:
        if control is not None:
            control.close()


def resume_repair(request: RepairRequest, proof: ProofIdentity) -> RepairReceipt:
    assert request.authority is not None
    root, journal_path = required_mutation_paths(request)
    journal = read_journal(
        journal_path,
        maintenance_root=root,
        authority=request.authority,
        state_db=request.state_db,
        clip_store=request.clip_store,
    )
    backup = _verify_journal_backup(journal, root, request)
    _verify_operation(journal, backup, request)
    if proof != journal.proof_identity:
        raise ClipConsistencyError("authority_drift", "quiescence proof identity changed")
    if journal.state == "UNKNOWN":
        raise ClipConsistencyError(
            "commit_state_unknown", "ambiguous commit state requires incident recovery"
        )
    if journal.state == "DONE":
        _validate_phase(request, journal, "held")
        validate_journal_identity(
            journal_path,
            request.authority,
            expected_sha256=journal_sha256(journal),
            expected_identity=journal.file_identity,
        )
        return journal_receipt("resume", journal)
    if journal.state == "ABORTED":
        _validate_phase(request, journal, "original")
        validate_journal_identity(
            journal_path,
            request.authority,
            expected_sha256=journal_sha256(journal),
            expected_identity=journal.file_identity,
        )
        return journal_receipt("resume", journal)
    _validate_phase(
        request,
        journal,
        "either" if journal.state == "PREPARED" else "held",
    )
    validate_journal_identity(
        journal_path,
        request.authority,
        expected_sha256=journal_sha256(journal),
        expected_identity=journal.file_identity,
    )
    control = open_write_exclusion(request.state_db)
    try:
        validate_database(control, now=time.time())
        if non_relation_preimage_sha256(control) != journal.non_relation_state_sha256:
            raise ClipConsistencyError("resume_conflict", "non-relation database preimage changed")
        current = relation_state_sha256(control)
        if journal.state == "PREPARED":
            journal = _resume_prepared(request, journal, control, current)
        else:
            if current != journal.relations_after_sha256:
                raise ClipConsistencyError("resume_conflict", "committed relation state differs")
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
    assert isinstance(control, sqlite3.Connection)
    assert request.authority is not None
    if current == journal.relations_before_sha256:
        ensure_quarantine(request.clip_store, journal.quarantine)
        _validate_phase(request, journal, "held")
        execute_plan(control, journal.relation_plan())
        try:
            _commit_connection(control)
        except BaseException as commit_error:
            control.close()
            moved = [
                (request.clip_store / original, request.clip_store / held)
                for original, held in journal.quarantine
            ]
            _classify_ambiguous_commit(request, journal, moved, commit_error)
            raise
        checkpoint(request.fault_hook, "resume:after_commit")
        _validate_phase(request, journal, "held")
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
        authority=request.authority,
        hook=request.fault_hook,
    )


def _verify_journal_backup(
    journal: ApplyJournal, root: Path, request: RepairRequest
) -> BackupReceipt:
    assert request.authority is not None
    receipt_path = Path(journal.backup_receipt_path)
    if sha256_regular(receipt_path) != journal.backup_receipt_sha256:
        raise ClipConsistencyError("backup_receipt_invalid", "journal receipt hash differs")
    backup = verify_backup_receipt_for_resume(
        receipt_path,
        maintenance_root=root,
        clip_store=request.clip_store,
        authority=request.authority,
    )
    if (
        backup.source_state_sha256 != journal.source_state_sha256
        or backup.source_identity_sha256 != journal.source_identity_sha256
        or backup.source_path != journal.state_db
    ):
        raise ClipConsistencyError("backup_receipt_invalid", "journal source identity differs")
    return backup


def _require_unchanged_manifest_authority(
    observed: ManifestAuthority, expected: ManifestAuthority
) -> None:
    if observed != expected:
        raise ClipConsistencyError("authority_drift", "manifest authority changed while planning")


def _is_security_drift(error: BaseException) -> bool:
    return isinstance(error, ClipConsistencyError) and error.code in {
        "authority_drift",
        "operation_digest_invalid",
        "backup_receipt_invalid",
        "backup_verification_failed",
    }


def _require_bound_proof(observed: ProofIdentity, expected: ProofIdentity) -> None:
    if observed != expected:
        raise ClipConsistencyError("operation_digest_invalid", "bound proof identity differs")


def _verify_operation(journal: ApplyJournal, backup: BackupReceipt, request: RepairRequest) -> None:
    assert request.authority is not None
    root, path = required_mutation_paths(request)
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
        journal_path=path,
        backup=backup,
        non_relation_state_sha256=journal.non_relation_state_sha256,
        plan=journal.relation_plan(),
        quarantine=journal.quarantine,
    )


def _validate_phase(request: RepairRequest, journal: ApplyJournal, state: str) -> None:
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
        quarantine_state=state,
        deleted_quarantine=journal.deleted_quarantine,
    )
    backup = _verify_journal_backup(journal, root, request)
    _verify_operation(journal, backup, request)


def _authority(request: RepairRequest) -> ManifestAuthority:
    assert request.authority is not None
    return scan_manifest_authority(
        request.clip_store,
        expected_uid=request.authority.clip_uid,
        expected_gid=request.authority.clip_gid,
        expected_dir_mode=request.authority.clip_dir_mode,
        ffprobe_bin=request.ffprobe_bin,
    )


__all__ = ["apply_repair", "resume_repair"]
