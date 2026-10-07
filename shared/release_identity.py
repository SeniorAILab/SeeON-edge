from __future__ import annotations

from typing import Final

EDGE_DATABASE_FORMAT_IDENTITY: Final = "seeon-edge-v1"
EDGE_DATABASE_SCHEMA_VERSION: Final = 19


class ReleaseIdentityMismatchError(RuntimeError):
    def __init__(self, found: int, expected: int = EDGE_DATABASE_SCHEMA_VERSION) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"edge database schema identity {found} does not match required {expected}"
        )


def require_peer_schema_identity(
    found: int,
    *,
    expected: int = EDGE_DATABASE_SCHEMA_VERSION,
) -> None:
    if found != expected:
        raise ReleaseIdentityMismatchError(found, expected)


__all__ = [
    "EDGE_DATABASE_FORMAT_IDENTITY",
    "EDGE_DATABASE_SCHEMA_VERSION",
    "ReleaseIdentityMismatchError",
    "require_peer_schema_identity",
]
