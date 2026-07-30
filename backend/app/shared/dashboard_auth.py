"""Server-validated dashboard sessions, separate from worker relay authority."""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

API_DASHBOARD_USERNAME_ENV = "API_DASHBOARD_USERNAME"
API_DASHBOARD_PASSWORD_ENV = "API_DASHBOARD_PASSWORD"
API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV = "API_ALLOW_LEGACY_DASHBOARD_AUTH"
API_EDGE_RELAY_TOKEN_ENV = "API_EDGE_RELAY_TOKEN"
DASHBOARD_SESSION_COOKIE = "ml_dashboard_session"
DASHBOARD_SESSION_TTL_SECONDS = 12 * 60 * 60
_SESSION_STORE_INIT_LOCK = threading.Lock()


@dataclass(slots=True)
class DashboardSessionStore:
    username: str
    password: str
    ttl_seconds: int = DASHBOARD_SESSION_TTL_SECONDS
    _sessions: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def authenticate(self, username: str, password: str) -> str | None:
        if not (
            hmac.compare_digest(username, self.username)
            and hmac.compare_digest(password, self.password)
        ):
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._prune_locked()
            self._sessions[token] = time.monotonic() + self.ttl_seconds
        return token

    def actor(self, token: str | None) -> str | None:
        if token is None:
            return None
        with self._lock:
            self._prune_locked()
            if token not in self._sessions:
                return None
        return self.username

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [token for token, deadline in self._sessions.items() if deadline <= now]
        for token in expired:
            self._sessions.pop(token, None)


def dashboard_sessions(request: Request) -> DashboardSessionStore | None:
    existing = getattr(request.app.state, "dashboard_sessions", None)
    if isinstance(existing, DashboardSessionStore):
        return existing
    with _SESSION_STORE_INIT_LOCK:
        existing = getattr(request.app.state, "dashboard_sessions", None)
        if isinstance(existing, DashboardSessionStore):
            return existing
        username = str(
            getattr(request.app.state, "dashboard_username", "")
            or os.environ.get(API_DASHBOARD_USERNAME_ENV, "")
        ).strip()
        password = str(
            getattr(request.app.state, "dashboard_password", "")
            or os.environ.get(API_DASHBOARD_PASSWORD_ENV, "")
        )
        if not username and not password:
            return None
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="dashboard credentials are incompletely configured",
            )
        store = DashboardSessionStore(username=username, password=password)
        request.app.state.dashboard_sessions = store
        return store


def authorize_dashboard(request: Request, *, legacy_token: str | None = None) -> str:
    """Authorize only a server-validated dashboard session."""

    sessions = dashboard_sessions(request)
    if sessions is not None:
        actor = sessions.actor(request.cookies.get(DASHBOARD_SESSION_COOKIE))
        if actor is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="dashboard session required",
            )
        return actor

    allow_legacy = os.environ.get(API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV, "").strip() == "1"
    if not allow_legacy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard authentication is not configured",
        )
    expected = getattr(request.app.state, "edge_relay_token", None) or os.environ.get(
        API_EDGE_RELAY_TOKEN_ENV
    )
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="legacy dashboard authentication is not configured",
        )
    if legacy_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="dashboard credential required",
        )
    if not hmac.compare_digest(legacy_token, str(expected)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="dashboard credential mismatch",
        )
    return "legacy-dashboard"


__all__ = [
    "API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV",
    "API_DASHBOARD_PASSWORD_ENV",
    "API_DASHBOARD_USERNAME_ENV",
    "DASHBOARD_SESSION_COOKIE",
    "DASHBOARD_SESSION_TTL_SECONDS",
    "DashboardSessionStore",
    "authorize_dashboard",
    "dashboard_sessions",
]
