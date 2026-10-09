from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class ArtifactReceiptVerificationError(RuntimeError): ...


class ArtifactDigest(Protocol):
    @property
    def sha256(self) -> str: ...

    @property
    def size_bytes(self) -> int: ...


def verify_artifact(path: Path, receipt: ArtifactDigest) -> None:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact is missing") from exc
    if not path.is_file() or stat_result.st_size != receipt.size_bytes:
        raise ArtifactReceiptVerificationError("artifact size does not match receipt")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact cannot be verified") from exc
    if digest.hexdigest() != receipt.sha256:
        raise ArtifactReceiptVerificationError("artifact hash does not match receipt")


__all__ = ["ArtifactDigest", "ArtifactReceiptVerificationError", "verify_artifact"]
