from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Final

from worker.adapters.model.errors import ModelLoadError

_DIGEST_LENGTH: Final = 64
_READ_CHUNK_BYTES: Final = 1024 * 1024


def artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelLoadError(f"cannot read model artifact: {path}") from exc
    return digest.hexdigest()


def verify_artifact_digest(path: Path, expected: str | None) -> str:
    actual = artifact_digest(path)
    if expected is None:
        return actual
    if (
        len(expected) != _DIGEST_LENGTH
        or expected.lower() != expected
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ModelLoadError("expected artifact digest must be lowercase SHA-256")
    if not hmac.compare_digest(actual, expected):
        raise ModelLoadError(f"artifact digest mismatch for {path.name}")
    return actual


__all__ = ["artifact_digest", "verify_artifact_digest"]
