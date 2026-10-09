"""Event-driven publication of complete edge topology snapshots."""

from __future__ import annotations

from typing import Protocol

from pydantic import JsonValue

from backend.app.features.audit.postgres_runtime import AuditMutation


class RosterSyncResult(Protocol):
    @property
    def status(self) -> str: ...

    @property
    def error_class(self) -> str | None: ...

    @property
    def detail(self) -> str | None: ...

    @property
    def last_ok_at(self) -> str | None: ...


class TopologyPublisher(Protocol):
    def trigger(
        self,
        *,
        force: bool = ...,
        refresh: bool = ...,
        now_epoch: float | None = ...,
        audit: AuditMutation | None = ...,
    ) -> RosterSyncResult: ...

    def current_result(self) -> RosterSyncResult: ...


def sync_camera_roster(
    coordinator: TopologyPublisher,
    *,
    _now: float | None = None,
    _force: bool = False,
    _refresh: bool = False,
    audit: AuditMutation | None = None,
) -> RosterSyncResult:
    """Publish at most one durable snapshot for this explicit event."""
    return coordinator.trigger(
        force=_force,
        refresh=_refresh,
        now_epoch=_now,
        audit=audit,
    )


def recover_camera_roster_on_boot(coordinator: TopologyPublisher) -> RosterSyncResult:
    """Resume one pending snapshot or recover one dirty registry snapshot."""
    return sync_camera_roster(coordinator, _force=True)


def resume_camera_roster_after_connectivity(
    coordinator: TopologyPublisher,
) -> RosterSyncResult:
    """Resume pending work after backend state and connectivity refresh."""
    return sync_camera_roster(coordinator, _force=True, _refresh=True)


def camera_sync_view(coordinator: TopologyPublisher, _camera_id: str) -> dict[str, JsonValue]:
    """Expose the durable complete-topology state through the legacy camera view."""
    result = coordinator.current_result()
    return {
        "status": result.status,
        "error_class": result.error_class,
        "detail": result.detail,
        "last_ok_at": result.last_ok_at,
    }


__all__ = [
    "RosterSyncResult",
    "TopologyPublisher",
    "camera_sync_view",
    "recover_camera_roster_on_boot",
    "resume_camera_roster_after_connectivity",
    "sync_camera_roster",
]
