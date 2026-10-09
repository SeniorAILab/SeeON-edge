from __future__ import annotations

from typing import Protocol

from backend.app.shared.dashboard_credentials import PersistedDashboardCredentials
from backend.app.shared.dashboard_sessions import DashboardSessionStore


class DashboardSessionRequired(Exception):
    ...


class PersistCredentials(Protocol):
    def __call__(self, username: str, password: str) -> PersistedDashboardCredentials: ...


def rotate_credentials(
    sessions: DashboardSessionStore,
    persist: PersistCredentials,
    *,
    token: str | None,
    new_username: str | None,
    new_password: str,
) -> str:
    if sessions.actor(token) is None:
        raise DashboardSessionRequired
    resolved_username = (new_username or "").strip() or sessions.username
    try:
        persisted = persist(resolved_username, new_password)
        return _mint_rotated_session(sessions, persisted, new_password)
    except BaseException:
        sessions.invalidate()
        raise


def _mint_rotated_session(
    sessions: DashboardSessionStore, persisted: PersistedDashboardCredentials, password: str
) -> str:
    sessions.rotate_credentials(persisted)
    token = sessions.authenticate(persisted.username, password)
    if token is None:
        raise RuntimeError("dashboard credential rotation produced an unauthenticated store")
    return token


__all__ = ["DashboardSessionRequired", "PersistCredentials", "rotate_credentials"]
