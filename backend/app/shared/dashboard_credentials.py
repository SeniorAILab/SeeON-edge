from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import UTC, datetime

_ALGORITHM_SCRYPT = "scrypt"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SALT_BYTES = 16


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )


@dataclass(frozen=True, slots=True)
class PersistedDashboardCredentials:
    username: str
    algorithm: str
    salt: bytes
    password_hash: bytes
    updated_at: str

    @classmethod
    def from_password(cls, *, username: str, password: str) -> PersistedDashboardCredentials:
        salt = os.urandom(_SALT_BYTES)
        return cls(
            username=username,
            algorithm=_ALGORITHM_SCRYPT,
            salt=salt,
            password_hash=_hash_password(password, salt),
            updated_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )

    def verify_password(self, password: str) -> bool:
        if self.algorithm != _ALGORITHM_SCRYPT:
            return False
        candidate = _hash_password(password, self.salt)
        return hmac.compare_digest(candidate, self.password_hash)


class DashboardCredentialsStoreError(RuntimeError):
    ...


__all__ = ["DashboardCredentialsStoreError", "PersistedDashboardCredentials"]
