from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from backend.app.shared.dashboard_credentials import PersistedDashboardCredentials

DASHBOARD_SESSION_TTL_SECONDS = 12 * 60 * 60


def _compare_str(candidate: str, expected: str) -> bool:
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


class DashboardCredentials(Protocol):
    @property
    def username(self) -> str: ...

    def verify(self, username: str, password: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlaintextDashboardCredentials:
    username: str
    password: str

    def verify(self, username: str, password: str) -> bool:
        return _compare_str(username, self.username) and _compare_str(password, self.password)


@dataclass(frozen=True, slots=True)
class HashedDashboardCredentials:
    persisted: PersistedDashboardCredentials

    @property
    def username(self) -> str:
        return self.persisted.username

    def verify(self, username: str, password: str) -> bool:
        return _compare_str(username, self.persisted.username) and (
            self.persisted.verify_password(password)
        )


@dataclass(slots=True)
class DashboardSessionStore:
    credentials: DashboardCredentials
    ttl_seconds: int = DASHBOARD_SESSION_TTL_SECONDS
    _sessions: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _active: bool = field(default=True, init=False, repr=False)

    @property
    def username(self) -> str:
        return self.credentials.username

    def authenticate(self, username: str, password: str) -> str | None:
        with self._lock:
            if not self._active or not self.credentials.verify(username, password):
                return None
            token = secrets.token_urlsafe(32)
            self._prune_locked()
            self._sessions[token] = time.monotonic() + self.ttl_seconds
        return token

    def actor(self, token: str | None) -> str | None:
        if token is None:
            return None
        with self._lock:
            self._prune_locked()
            if not self._active or token not in self._sessions:
                return None
            return self.credentials.username

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def invalidate(self) -> None:
        with self._lock:
            self._active = False
            self._sessions.clear()

    def rotate_credentials(self, persisted: PersistedDashboardCredentials) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("dashboard session store is inactive")
            self.credentials = HashedDashboardCredentials(persisted)
            self._sessions.clear()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [token for token, deadline in self._sessions.items() if deadline <= now]
        for token in expired:
            self._sessions.pop(token, None)


__all__ = [
    "DASHBOARD_SESSION_TTL_SECONDS",
    "DashboardCredentials",
    "DashboardSessionStore",
    "HashedDashboardCredentials",
    "PlaintextDashboardCredentials",
]
