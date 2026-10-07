from __future__ import annotations

from collections.abc import Callable

from backend.app.shared.dashboard_credentials import PersistedDashboardCredentials
from backend.app.shared.dashboard_sessions import DashboardSessionStore
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore


class DashboardSessionRequired(Exception):
    ...


def rotate_credentials(
    sessions: DashboardSessionStore,
    store: PostgresDashboardCredentialsStore,
    *,
    token: str | None,
    new_username: str | None,
    new_password: str,
    persist: Callable[[PostgresDashboardCredentialsStore, str, str], PersistedDashboardCredentials],
) -> str:
    if sessions.actor(token) is None:
        raise DashboardSessionRequired
    resolved_username = (new_username or "").strip() or sessions.username
    try:
        persisted = persist(store, resolved_username, new_password)
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


__all__ = ["DashboardSessionRequired", "rotate_credentials"]
