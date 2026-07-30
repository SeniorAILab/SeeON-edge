"""Runtime status route.

``/status`` merges two API-owned relay snapshots: heartbeat-derived camera
liveness and worker-published runtime diagnostics. It never reads worker runtime
state directly (no cross-process shared state).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.features.status.runtime_status_store import get_runtime_status_store

router = APIRouter(tags=["status"])


@router.get("/status")
def status(request: Request) -> dict[str, object]:
    heartbeat_store = get_heartbeat_store(request.app)
    inventory = getattr(request.app.state, "camera_inventory", {})
    response = heartbeat_store.snapshot(inventory)
    response["runtime"] = get_runtime_status_store(request.app).snapshot()
    return response


__all__ = ["router"]
