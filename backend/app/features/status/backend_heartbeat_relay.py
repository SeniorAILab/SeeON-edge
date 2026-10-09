from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.shared.http.backend_client_bundle import backend_client_bundle

logger = logging.getLogger(__name__)

MAX_BACKOFF_MULTIPLIER = 8

_ERROR_CLASS_SEVERITY: dict[str, int] = {"auth": 3, "timeout": 2, "unreachable": 1}


@dataclass(slots=True)
class RelayTickResult:
    attempted: int = 0
    sent: int = 0
    failed: int = 0
    skipped_reason: str | None = None
    error_class: str | None = None


@dataclass(slots=True)
class HeartbeatRelayState:
    consecutive_all_fail_ticks: int = 0
    backoff_multiplier: int = 1
    last_error_class: str | None = None
    last_success_at: str | None = None


def get_heartbeat_relay_state(app: object) -> HeartbeatRelayState:
    state = app.state  # type: ignore[attr-defined]
    relay_state = getattr(state, "backend_heartbeat_relay_state", None)
    if not isinstance(relay_state, HeartbeatRelayState):
        relay_state = HeartbeatRelayState()
        state.backend_heartbeat_relay_state = relay_state
    return relay_state


def effective_relay_interval_sec(
    base_interval_sec: float, relay_state: HeartbeatRelayState
) -> float:
    return base_interval_sec * relay_state.backoff_multiplier


def relay_heartbeats_once(app: object, now: float | None = None) -> RelayTickResult:
    bundle = backend_client_bundle(app)
    client = (
        bundle.ingest_client
        if bundle is not None
        else getattr(app.state, "backend_ingest_client", None)
    )
    if client is None or not hasattr(client, "for_camera"):
        return RelayTickResult(skipped_reason="no_client")

    from backend.app.features.cameras.store import CameraRegistryStore, registry_expected_cameras

    registry = getattr(app.state, "camera_registry", None)
    registry_store = registry if isinstance(registry, CameraRegistryStore) else None
    expected = registry_expected_cameras(registry_store)
    snapshot = get_heartbeat_store(app).snapshot(expected, now=now)
    online_camera_ids = [
        camera_id for camera_id, info in snapshot["cameras"].items() if info["status"] == ONLINE
    ]
    if not online_camera_ids:
        return RelayTickResult(skipped_reason="no_online_cameras")

    canonical_camera_ids: list[str] = []
    for camera_id in online_camera_ids:
        canonical_id = _canonical_backend_camera_id(registry_store, camera_id)
        if canonical_id is None:
            logger.info(
                "backend heartbeat relay: skipping camera_id=%s, no backend mapping yet",
                camera_id,
            )
            continue
        canonical_camera_ids.append(canonical_id)

    if not canonical_camera_ids:
        return RelayTickResult(skipped_reason="no_mapped_cameras")

    result = RelayTickResult(attempted=len(canonical_camera_ids))
    for camera_id in canonical_camera_ids:
        ok, error_class = _send_one(client, camera_id)
        if ok:
            result.sent += 1
        else:
            result.failed += 1
            result.error_class = _more_severe(result.error_class, error_class)

    relay_state = get_heartbeat_relay_state(app)
    _update_backoff(relay_state, result)
    _update_error_state(relay_state, result)
    return result


def _canonical_backend_camera_id(registry: object | None, camera_id: str) -> str | None:
    if registry is None:
        return None
    snapshot = registry.snapshot()  # type: ignore[attr-defined]
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return None
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        if camera_id in (local_id, backend_id):
            return backend_id if isinstance(backend_id, str) and backend_id else None
    return None


def _send_one(client: object, camera_id: str) -> tuple[bool, str | None]:
    try:
        camera_client = client.for_camera(camera_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False, None
    result_sender = getattr(camera_client, "send_heartbeat_result", None)
    if callable(result_sender):
        try:
            sent = result_sender()
        except Exception:  # noqa: BLE001
            return False, None
        return bool(getattr(sent, "ok", False)), getattr(sent, "error_class", None)
    sender = getattr(camera_client, "send_heartbeat", None)
    if not callable(sender):
        return False, None
    try:
        return bool(sender()), None
    except Exception:  # noqa: BLE001
        return False, None


def _more_severe(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if _ERROR_CLASS_SEVERITY.get(candidate, 0) > _ERROR_CLASS_SEVERITY.get(current, 0):
        return candidate
    return current


def _update_backoff(relay_state: HeartbeatRelayState, result: RelayTickResult) -> None:
    if result.attempted == 0:
        return
    if result.failed == result.attempted:
        relay_state.consecutive_all_fail_ticks += 1
        relay_state.backoff_multiplier = min(
            MAX_BACKOFF_MULTIPLIER, relay_state.backoff_multiplier * 2
        )
    else:
        relay_state.consecutive_all_fail_ticks = 0
        relay_state.backoff_multiplier = 1


def _update_error_state(relay_state: HeartbeatRelayState, result: RelayTickResult) -> None:
    if result.sent > 0:
        relay_state.last_success_at = _utc_now_iso()
    if result.attempted == 0:
        return
    previous = relay_state.last_error_class
    relay_state.last_error_class = result.error_class
    if result.error_class != previous:
        logger.warning(
            "backend heartbeat relay error_class transition: %s -> %s",
            previous,
            result.error_class,
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "MAX_BACKOFF_MULTIPLIER",
    "HeartbeatRelayState",
    "RelayTickResult",
    "effective_relay_interval_sec",
    "get_heartbeat_relay_state",
    "relay_heartbeats_once",
]
