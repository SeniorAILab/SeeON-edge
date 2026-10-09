from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

from backend.app.shared.artifact_verification import ArtifactReceiptVerificationError

if TYPE_CHECKING:
    from backend.app.features.clips.manifest import ClipManifest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactReceiptConflictError(RuntimeError):
    ...


class ArtifactReceiptPersistenceError(RuntimeError):
    ...


class ReceiptMissingIncidentError(ArtifactReceiptPersistenceError):
    ...


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    artifact_id: str
    sha256: str
    size_bytes: int
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.artifact_id or "\x00" in self.artifact_id:
            raise ValueError("invalid artifact identity")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("invalid artifact hash")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("invalid artifact size")


@runtime_checkable
class ArtifactReceiptStore(Protocol):
    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt: ...

    def get(self, artifact_id: str) -> ArtifactReceipt | None: ...


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    handle: BinaryIO
    sha256: str
    size_bytes: int
    device: int
    inode: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True, slots=True)
class ClipProjection:
    receipt: ArtifactReceipt
    verified: VerifiedArtifact
    manifest: ClipManifest
    manifest_relpath: str
    media_relpath: str
    manifest_hash: str
    manifest_size: int


def primary_artifact_id(clip_id: str, edge_event_id: str) -> str:
    digest = hashlib.sha256(f"{clip_id}\x1f{edge_event_id}".encode()).hexdigest()[:32]
    return f"primary:{digest}"


def verified_artifact(handle: BinaryIO) -> VerifiedArtifact:
    try:
        descriptor_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise ArtifactReceiptVerificationError("artifact descriptor is not regular")
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
    except OSError as error:
        raise ArtifactReceiptVerificationError("artifact descriptor cannot be verified") from error
    return VerifiedArtifact(
        handle=handle,
        sha256=digest.hexdigest(),
        size_bytes=descriptor_stat.st_size,
        device=descriptor_stat.st_dev,
        inode=descriptor_stat.st_ino,
    )


__all__ = [
    "ArtifactReceipt",
    "ArtifactReceiptConflictError",
    "ArtifactReceiptPersistenceError",
    "ArtifactReceiptStore",
    "ClipProjection",
    "ReceiptMissingIncidentError",
    "VerifiedArtifact",
    "primary_artifact_id",
    "verified_artifact",
]
