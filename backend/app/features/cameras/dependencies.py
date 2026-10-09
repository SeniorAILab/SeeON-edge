"""Event-driven publication of complete edge topology snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import JsonValue

from backend.app.features.audit.postgres_runtime import AuditMutation
from backend.app.features.cameras.roster_sync import RosterSyncResult, TopologyPublisher
from backend.app.features.cameras.roster_sync import (
    camera_sync_view as sync_view,
)
from backend.app.features.cameras.roster_sync import (
    sync_camera_roster as sync_roster,
)
from backend.app.features.cameras.store import CameraRegistryStore


@dataclass(frozen=True, slots=True)
class CameraPorts:
    enrolled_facility_id: Callable[[], str | None]
    topology: Callable[[], TopologyPublisher]
    heartbeats: Callable[[], dict[str, object]]


def camera_ports(app: FastAPI) -> CameraPorts:
    ports = getattr(app.state, "camera_ports", None)
    if ports is None:
        raise RuntimeError("camera ports are not injected")
    if not isinstance(ports, CameraPorts):
        raise TypeError("camera ports have invalid type")
    return ports


def camera_registry(app: FastAPI) -> CameraRegistryStore:
    store = getattr(app.state, "camera_registry", None)
    if store is None:
        raise RuntimeError("camera registry is not injected")
    if not isinstance(store, CameraRegistryStore):
        raise TypeError("camera registry has invalid type")
    return store


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
        camera_ports(app).topology(),
        _force=_force,
        _refresh=_refresh,
        _now=_now,
        audit=audit,
    )


def camera_sync_view(app: FastAPI, _camera_id: str) -> dict[str, JsonValue]:
    """Expose the durable complete-topology state through the legacy camera view."""
    return sync_view(camera_ports(app).topology(), _camera_id)


__all__ = [
    "CameraPorts",
    "RosterSyncResult",
    "camera_ports",
    "camera_registry",
    "camera_sync_view",
    "sync_camera_roster",
]
