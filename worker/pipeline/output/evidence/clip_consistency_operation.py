"""Canonical versioned digest for one clip-consistency mutation operation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_authority_types import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_database import RelationPlan
from worker.pipeline.output.evidence.clip_consistency_phase_authority import (
    AuthoritySnapshot,
    FileIdentity,
    ProofIdentity,
    identity_for_path,
)
from worker.pipeline.output.evidence.clip_consistency_quarantine import (
    canonical_quarantine_rows,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    BackupReceipt,
    ClipConsistencyError,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import SCHEMA_VERSION

OPERATION_DIGEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class OperationBinding:
    version: int
    digest: str
    quarantine_namespace_sha256: str
    image_artifact_identity: str
    snapshot: AuthoritySnapshot
    proof: ProofIdentity
    backup_receipt_file_identity: FileIdentity
    maintenance_root: str


def build_operation_binding(
    *,
    authority: RepairAuthority,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    journal_path: Path,
    proof: ProofIdentity,
    backup: BackupReceipt,
    snapshot: AuthoritySnapshot,
    non_relation_state_sha256: str,
    plan: RelationPlan,
) -> tuple[OperationBinding, tuple[tuple[str, str], ...]]:
    image_identity = image_artifact_identity(authority)
    receipt_file_identity = identity_for_path(Path(backup.receipt_path))
    base = _base_payload(
        authority=authority,
        state_db=state_db,
        clip_store=clip_store,
        maintenance_root=maintenance_root,
        journal_path=journal_path,
        proof=proof,
        backup=backup,
        receipt_file_identity=receipt_file_identity,
        snapshot=snapshot,
        image_identity=image_identity,
        non_relation_state_sha256=non_relation_state_sha256,
        plan=plan,
    )
    namespace = _sha256({"operation_namespace_version": 1, **base})
    rows = canonical_quarantine_rows(plan.quarantine_clip_ids, namespace)
    digest = _sha256(
        {
            "operation_digest_version": OPERATION_DIGEST_VERSION,
            **base,
            "quarantine_namespace_sha256": namespace,
            "quarantine": [list(row) for row in rows],
        }
    )
    bound_proof = replace(proof, operation_digest=digest)
    return (
        OperationBinding(
            version=OPERATION_DIGEST_VERSION,
            digest=digest,
            quarantine_namespace_sha256=namespace,
            image_artifact_identity=image_identity,
            snapshot=snapshot,
            proof=bound_proof,
            backup_receipt_file_identity=receipt_file_identity,
            maintenance_root=str(maintenance_root.resolve(strict=True)),
        ),
        rows,
    )


def verify_operation_binding(
    *,
    version: int,
    digest: str,
    quarantine_namespace_sha256: str,
    image_identity: str,
    snapshot: AuthoritySnapshot,
    proof: ProofIdentity,
    maintenance_root_path: str,
    authority: RepairAuthority,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    journal_path: Path,
    backup: BackupReceipt,
    non_relation_state_sha256: str,
    plan: RelationPlan,
    quarantine: tuple[tuple[str, str], ...],
) -> None:
    binding, expected_rows = build_operation_binding(
        authority=authority,
        state_db=state_db,
        clip_store=clip_store,
        maintenance_root=maintenance_root,
        journal_path=journal_path,
        proof=proof,
        backup=backup,
        snapshot=snapshot,
        non_relation_state_sha256=non_relation_state_sha256,
        plan=plan,
    )
    valid = (
        version == binding.version
        and digest == binding.digest
        and quarantine_namespace_sha256 == binding.quarantine_namespace_sha256
        and image_identity == binding.image_artifact_identity
        and maintenance_root_path == binding.maintenance_root
        and quarantine == expected_rows
        and backup.operation_digest_version == version
        and backup.operation_digest == digest
    )
    if not valid:
        raise ClipConsistencyError("operation_digest_invalid", "operation binding differs")


def image_artifact_identity(authority: RepairAuthority) -> str:
    packaged = os.environ.get("CLIP_CONSISTENCY_TOOL_REVISION", authority.tool_revision)
    return _sha256(
        {
            "artifact_contract": "clip-consistency-tool-image-v1",
            "packaged_revision": packaged,
            "requested_revision": authority.tool_revision,
        }
    )


def receipt_semantic_payload(receipt: BackupReceipt) -> dict[str, object]:
    payload = asdict(receipt)
    payload.pop("operation_digest_version", None)
    payload.pop("operation_digest", None)
    return payload


def _base_payload(
    *,
    authority: RepairAuthority,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    journal_path: Path,
    proof: ProofIdentity,
    backup: BackupReceipt,
    receipt_file_identity: FileIdentity,
    snapshot: AuthoritySnapshot,
    image_identity: str,
    non_relation_state_sha256: str,
    plan: RelationPlan,
) -> dict[str, object]:
    return {
        "paths": {
            "state_db": str(state_db.resolve(strict=True)),
            "clip_store": str(clip_store.resolve(strict=True)),
            "maintenance_root": str(maintenance_root.resolve(strict=True)),
            "journal": str(journal_path.resolve(strict=False)),
        },
        "authority": authority.to_dict(),
        "authority_sha256": authority.sha256,
        "image_artifact_identity": image_identity,
        "proof_identity": proof.semantic_dict(),
        "source_backup_receipt_identity": receipt_semantic_payload(backup),
        "backup_receipt_file_identity": receipt_file_identity.to_dict(),
        "authority_snapshot": snapshot.to_dict(),
        "schema_version": SCHEMA_VERSION,
        "non_relation_state_sha256": non_relation_state_sha256,
        "source_state_sha256": backup.source_state_sha256,
        "source_identity_sha256": backup.source_identity_sha256,
        "relation_plan": plan.authority_payload(),
        "plan_sha256": plan.plan_sha256,
        "relations_before_sha256": plan.before_sha256,
        "relations_after_sha256": plan.after_sha256,
    }


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "OPERATION_DIGEST_VERSION",
    "OperationBinding",
    "build_operation_binding",
    "image_artifact_identity",
    "receipt_semantic_payload",
    "verify_operation_binding",
]
