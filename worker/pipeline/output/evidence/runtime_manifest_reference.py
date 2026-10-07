from __future__ import annotations

from enum import StrEnum


class RuntimeManifestReferenceFailure(StrEnum):
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class RuntimeManifestReferenceError(RuntimeError):
    def __init__(
        self,
        manifest_sha256: str,
        failure: RuntimeManifestReferenceFailure,
    ) -> None:
        self.manifest_sha256 = manifest_sha256
        self.failure = failure
        super().__init__(f"runtime manifest reference {manifest_sha256} is {failure.value}")


__all__ = [
    "RuntimeManifestReferenceError",
    "RuntimeManifestReferenceFailure",
]
