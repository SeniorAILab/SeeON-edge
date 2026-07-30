from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from contracts.worker_config import WORKER_CONFIG_PATH, PulledCameraConfig, PulledWorkerConfig
from edge.runtime.edge_worker_config import (
    CameraRuntimeConfig,
    DomainsConfig,
    EdgeWorkerConfig,
    RelayConfig,
)

ML_WORKER_STATE_DIR_ENV = "ML_WORKER_STATE_DIR"
_DEFAULT_STATE_DIR = "/var/lib/ml-worker"
_LKG_FILENAME = "worker-config-last-known-good.json"


def pull_worker_config(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
) -> PulledWorkerConfig | None:
    raw = _pull_worker_config_payload(relay_url, relay_token, timeout_sec=timeout_sec)
    if raw is None:
        return None
    try:
        return _pulled_from_payload(raw)
    except (ValueError, TypeError) as exc:
        print(f"worker config pull skipped: {exc}", file=sys.stderr)
        return None


def pull_edge_worker_config(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
) -> tuple[EdgeWorkerConfig, int] | None:
    raw = _pull_worker_config_payload(relay_url, relay_token, timeout_sec=timeout_sec)
    if raw is None:
        return None
    try:
        config, registry_version = edge_worker_config_from_payload(raw, relay_url, relay_token)
    except (ValueError, TypeError) as exc:
        print(f"worker config pull skipped: {exc}", file=sys.stderr)
        return None
    save_worker_config_lkg(raw)
    return config, registry_version


def load_edge_worker_config_from_relay(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
) -> tuple[EdgeWorkerConfig, int, str] | None:
    pulled = pull_edge_worker_config(relay_url, relay_token, timeout_sec=timeout_sec)
    if pulled is not None:
        config, registry_version = pulled
        return config, registry_version, "pulled"
    raw = load_worker_config_lkg()
    if raw is None:
        return None
    try:
        config, registry_version = edge_worker_config_from_payload(raw, relay_url, relay_token)
    except (ValueError, TypeError) as exc:
        print(f"worker config LKG skipped: {exc}", file=sys.stderr)
        return None
    return config, registry_version, "lkg"


def edge_worker_config_from_payload(
    payload: dict[str, object],
    relay_url: str,
    relay_token: str | None,
) -> tuple[EdgeWorkerConfig, int]:
    registry_version = _registry_version(payload)
    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, list):
        raise TypeError("cameras must be a list")
    cameras = tuple(_camera_runtime_config(item) for item in raw_cameras if isinstance(item, dict))
    if not cameras:
        raise ValueError("worker config must include at least one camera")
    domain_names = _domain_union(raw_cameras)
    config = EdgeWorkerConfig(
        relay=RelayConfig(url=relay_url, token=_required_token(relay_token)),
        domains=DomainsConfig(enabled=domain_names or None),
        cameras=cameras,
    )
    return config, registry_version


def save_worker_config_lkg(payload: dict[str, object]) -> None:
    path = worker_config_lkg_path()
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"worker config LKG save failed: {exc}", file=sys.stderr)


def load_worker_config_lkg() -> dict[str, object] | None:
    try:
        loaded = json.loads(worker_config_lkg_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"worker config LKG load skipped: {exc}", file=sys.stderr)
        return None
    if not isinstance(loaded, dict):
        print("worker config LKG load skipped: payload must be an object", file=sys.stderr)
        return None
    return loaded


def worker_config_lkg_path() -> Path:
    return Path(os.environ.get(ML_WORKER_STATE_DIR_ENV, _DEFAULT_STATE_DIR)) / _LKG_FILENAME


def _pull_worker_config_payload(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float,
) -> dict[str, object] | None:
    url = f"{relay_url.rstrip('/')}{WORKER_CONFIG_PATH}"
    headers = {"Accept": "application/json"}
    if relay_token:
        headers["X-Edge-Relay-Token"] = relay_token
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                print(f"worker config pull skipped: HTTP {status}", file=sys.stderr)
                return None
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
    except (
        TimeoutError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
    ) as exc:
        print(f"worker config pull skipped: {exc}", file=sys.stderr)
        return None
    if not isinstance(parsed, dict):
        print("worker config pull skipped: payload must be an object", file=sys.stderr)
        return None
    return parsed


def _pulled_from_payload(payload: dict[str, object]) -> PulledWorkerConfig:
    if "registry_version" not in payload:
        return PulledWorkerConfig.from_dict(payload)
    registry_version = _registry_version(payload)
    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, list):
        raise TypeError("cameras must be a list")
    return PulledWorkerConfig(
        config_version=registry_version,
        restart_epoch=registry_version,
        night_window=None,
        cameras=tuple(
            PulledCameraConfig(
                camera_id=_require_str(item, "camera_id"),
                space_id=_require_str(item, "facility_id"),
                label=_optional_str(item.get("label")) or _require_str(item, "camera_id"),
                rtsp_url=_require_str(item, "rtsp_url"),
                online=True,
            )
            for item in raw_cameras
            if isinstance(item, dict)
        ),
    )


def _camera_runtime_config(data: dict[str, object]) -> CameraRuntimeConfig:
    kwargs: dict[str, Any] = {
        "camera_id": _require_str(data, "camera_id"),
        "facility_id": _require_str(data, "facility_id"),
        "rtsp_url": _require_str(data, "rtsp_url"),
    }
    fps = data.get("fps")
    if isinstance(fps, int | float) and not isinstance(fps, bool):
        kwargs["fps"] = float(fps)
    decode_backend = data.get("decode_backend")
    if isinstance(decode_backend, str) and decode_backend.strip():
        kwargs["decode_backend"] = decode_backend
    return CameraRuntimeConfig(**kwargs)


def _domain_union(raw_cameras: list[object]) -> tuple[str, ...]:
    domains: set[str] = set()
    for item in raw_cameras:
        if not isinstance(item, dict):
            continue
        raw_domains = item.get("domains")
        if raw_domains is None:
            continue
        if not isinstance(raw_domains, list):
            raise TypeError("camera domains must be a list")
        domains.update(_require_domain(value) for value in raw_domains)
    return tuple(sorted(domains))


def _require_domain(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("camera domains must be non-empty strings")
    return value.strip()


def _registry_version(payload: dict[str, object]) -> int:
    value = payload.get("registry_version", payload.get("config_version"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("registry_version must be an integer")
    return value


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be a string or null")
    stripped = value.strip()
    return stripped or None


def _required_token(relay_token: str | None) -> str:
    if relay_token is None or not relay_token.strip():
        raise ValueError("RELAY_TOKEN is required for pulled worker config")
    return relay_token.strip()


__all__ = [
    "WORKER_CONFIG_PATH",
    "edge_worker_config_from_payload",
    "load_edge_worker_config_from_relay",
    "load_worker_config_lkg",
    "pull_edge_worker_config",
    "pull_worker_config",
    "save_worker_config_lkg",
    "worker_config_lkg_path",
]
