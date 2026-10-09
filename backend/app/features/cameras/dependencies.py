"""Event-driven publication of complete edge topology snapshots."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import JsonValue

from backend.app.features.audit.postgres_runtime import AuditMutation
from backend.app.features.cameras.roster_sync import RosterSyncResult
from backend.app.features.cameras.roster_sync import (
    camera_sync_view as sync_view,
)
from backend.app.features.cameras.roster_sync import (
    sync_camera_roster as sync_roster,
)
from backend.app.features.connection.dependencies import topology_retry_coordinator


def sync_camera_roster(
    app: FastAPI,
    *,
    _now: float | None = None,
    _force: bool = False,
    _refresh: bool = False,
    audit: AuditMutation | None = None,
) -> RosterSyncResult:
    """Publish at most one durable snapshot for this explicit event."""
    return sync_roster(
        topology_retry_coordinator(app),
        _force=_force,
        _refresh=_refresh,
        _now=_now,
        audit=audit,
    )


def camera_sync_view(app: FastAPI, _camera_id: str) -> dict[str, JsonValue]:
    """Expose the durable complete-topology state through the legacy camera view."""
    return sync_view(topology_retry_coordinator(app), _camera_id)


__all__ = [
    "RosterSyncResult",
    "camera_sync_view",
    "sync_camera_roster",
]
