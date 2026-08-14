"""Strict operator assertion that every worker-state writer is stopped."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from worker.pipeline.output.evidence.clip_consistency_authority import (
    AUTHORITY_KEYS,
    RepairAuthority,
)
from worker.pipeline.output.evidence.clip_consistency_io import (
    read_strict_json,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_operation import (
    OPERATION_DIGEST_VERSION,
    image_artifact_identity,
)
from worker.pipeline.output.evidence.clip_consistency_phase_authority import (
    ProofIdentity,
    capture_proof_identity,
)
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError

_KEYS = frozenset(
    {
        "format_version",
        "state_db",
        "clip_store",
        "stopped_service",
        "stopped_db_writers",
        "operator_uid",
        "authority_sha256",
        "operation_digest_version",
        "operation_digest",
        "image_artifact_identity",
        "issued_at",
        "expires_at",
    }
    | AUTHORITY_KEYS
)
_WRITERS = ("event", "config", "fault")


def validate_quiescence_receipt(
    path: Path,
    *,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    authority: RepairAuthority,
    now: float | None = None,
) -> ProofIdentity:
    validate_under_root(path, maintenance_root, allow_missing_leaf=False)
    try:
        before = capture_proof_identity(path, authority)
    except ClipConsistencyError as exc:
        raise ClipConsistencyError(
            "quiescence_invalid", "receipt file authority differs"
        ) from exc
    payload = read_strict_json(
        path,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        exact_mode=0o600,
        error_code="quiescence_invalid",
    )
    if frozenset(payload) != _KEYS:
        raise ClipConsistencyError("quiescence_invalid", "receipt key set differs")
    issued_at = _integer(payload, "issued_at")
    expires_at = _integer(payload, "expires_at")
    observed = time.time() if now is None else now
    writers = payload.get("stopped_db_writers")
    valid = (
        _integer(payload, "format_version") == 3
        and _integer(payload, "operator_uid") == authority.state_uid
        and _receipt_authority(payload) == authority
        and _string(payload, "authority_sha256") == authority.sha256
        and _integer(payload, "operation_digest_version") == OPERATION_DIGEST_VERSION
        and _is_sha256(_string(payload, "operation_digest"))
        and _string(payload, "image_artifact_identity")
        == image_artifact_identity(authority)
        and _string(payload, "state_db") == str(state_db.resolve(strict=True))
        and _string(payload, "clip_store") == str(clip_store.resolve(strict=True))
        and _string(payload, "stopped_service") == "ml-worker"
        and isinstance(writers, list)
        and tuple(cast("list[object]", writers)) == _WRITERS
        and issued_at <= observed <= expires_at
        and 0 < expires_at - issued_at <= 3600
    )
    if not valid:
        raise ClipConsistencyError("quiescence_invalid", "receipt assertion differs")
    after = capture_proof_identity(path, authority)
    if after != before:
        raise ClipConsistencyError("quiescence_invalid", "receipt identity changed while read")
    return after


def bind_quiescence_receipt(
    path: Path,
    operation_digest: str,
    *,
    authority: RepairAuthority,
    expected_identity: ProofIdentity,
) -> ProofIdentity:
    if not _is_sha256(operation_digest):
        raise ClipConsistencyError("operation_digest_invalid", "operation digest is invalid")
    payload = read_strict_json(
        path,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        exact_mode=0o600,
        error_code="quiescence_invalid",
    )
    if frozenset(payload) != _KEYS:
        raise ClipConsistencyError("quiescence_invalid", "receipt key set differs")
    payload["operation_digest"] = operation_digest
    descriptor = os.open(path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(descriptor)
    expected = expected_identity.file
    if (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode & 0o7777,
    ) != (
        expected.device,
        expected.inode,
        expected.uid,
        expected.gid,
        expected.mode,
    ):
        os.close(descriptor)
        raise ClipConsistencyError(
            "authority_drift", "quiescence proof changed before binding"
        )
    os.ftruncate(descriptor, 0)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return capture_proof_identity(path, authority)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _receipt_authority(payload: Mapping[str, object]) -> RepairAuthority:
    return RepairAuthority(
        state_uid=_integer(payload, "state_uid"),
        state_gid=_integer(payload, "state_gid"),
        state_db_mode=_integer(payload, "state_db_mode"),
        state_dir_mode=_integer(payload, "state_dir_mode"),
        clip_uid=_integer(payload, "clip_uid"),
        clip_gid=_integer(payload, "clip_gid"),
        clip_dir_mode=_integer(payload, "clip_dir_mode"),
        tool_revision=_string(payload, "tool_revision"),
    )


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClipConsistencyError("quiescence_invalid", "receipt field type differs")
    return value


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ClipConsistencyError("quiescence_invalid", "receipt field type differs")
    return value


__all__ = ["bind_quiescence_receipt", "validate_quiescence_receipt"]
