"""Local camera registry storage and lightweight probes for ml-api."""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

API_CAMERA_STORE_ENV = "API_CAMERA_STORE"
DEFAULT_CAMERA_STORE = "/var/lib/ml-api/cameras.json"

CameraStatus = Literal["online", "offline", "starting", "unknown"]
ProbeErrorClass = Literal["timeout", "decode", "auth"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    error_class: ProbeErrorClass | None = None
    width: int | None = None
    height: int | None = None


class CameraRegistryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> CameraRegistryStore:
        return cls(os.environ.get(API_CAMERA_STORE_ENV, DEFAULT_CAMERA_STORE))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._read_unlocked()

    def create(
        self,
        *,
        camera_id: str | None = None,
        label: str,
        rtsp_url: str,
        space_id: str | None,
        status: CameraStatus,
        backend_camera_id: str | None = None,
        mapping_pending: bool = False,
        decode_backend: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            data = self._read_unlocked()
            record = {
                "id": camera_id or str(uuid.uuid4()),
                "label": label,
                "rtsp_url": rtsp_url,
                "space_id": space_id,
                "backend_camera_id": backend_camera_id,
                "mapping_pending": mapping_pending,
                "status": status,
                "decode_backend": decode_backend,
                "created_at": _utc_now(),
            }
            data["cameras"].append(record)
            data["registry_version"] += 1
            self._write_unlocked(data)
            return dict(record)

    def update(self, camera_id: str, updates: dict[str, object]) -> dict[str, object] | None:
        with self._lock:
            data = self._read_unlocked()
            for index, record in enumerate(data["cameras"]):
                if record.get("id") != camera_id:
                    continue
                updated = {**record, **updates}
                data["cameras"][index] = updated
                data["registry_version"] += 1
                self._write_unlocked(data)
                return dict(updated)
            return None

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            data = self._read_unlocked()
            cameras = data["cameras"]
            kept = [record for record in cameras if record.get("id") != camera_id]
            if len(kept) == len(cameras):
                return False
            data["cameras"] = kept
            data["registry_version"] += 1
            self._write_unlocked(data)
            return True

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            data = self._read_unlocked()
            for record in data["cameras"]:
                if record.get("id") == camera_id:
                    return dict(record)
            return None

    def _read_unlocked(self) -> dict[str, object]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"registry_version": 0, "cameras": []}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"registry_version": 0, "cameras": []}
        if not isinstance(raw, dict):
            return {"registry_version": 0, "cameras": []}
        registry_version = raw.get("registry_version", 0)
        if isinstance(registry_version, bool) or not isinstance(registry_version, int):
            registry_version = 0
        cameras = raw.get("cameras", [])
        if not isinstance(cameras, list):
            cameras = []
        return {
            "registry_version": max(0, registry_version),
            "cameras": [record for record in cameras if isinstance(record, dict)],
        }

    def _write_unlocked(self, data: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        # RTSP credentials are stored in local edge JSON by design; API responses
        # and logs must use mask_rtsp_url(), and the store is best-effort 0600.
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


def public_camera(record: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(record.get("id", "")),
        "label": str(record.get("label", "")),
        "rtsp_url_masked": mask_rtsp_url(str(record.get("rtsp_url", ""))),
        "space_id": _optional_str(record.get("space_id")),
        "backend_camera_id": _optional_str(record.get("backend_camera_id")),
        "status": _status(record.get("status")),
        "decode_backend": _optional_str(record.get("decode_backend")),
        "created_at": str(record.get("created_at", "")),
    }


def mask_rtsp_url(rtsp_url: str) -> str:
    try:
        parsed = urlsplit(rtsp_url)
        port = parsed.port
    except ValueError:
        return "RTSP URL masked"
    if not parsed.hostname:
        return "RTSP URL masked"
    host = "redacted-camera"
    if port is not None:
        host = f"{host}:{port}"
    if parsed.username is None and parsed.password is None:
        return urlunsplit(parsed._replace(netloc=host))
    user = "***" if parsed.username is not None else ""
    password = ":***" if parsed.password is not None else ""
    return urlunsplit(parsed._replace(netloc=f"{user}{password}@{host}"))


def probe_rtsp_url(rtsp_url: str, *, timeout_sec: float = 1.0) -> ProbeResult:
    """Perform an API-safe RTSP reachability probe.

    ml-api is intentionally ML-free and does not import cv2 or decode frames here.
    This probe only checks whether the RTSP TCP endpoint accepts a connection;
    deeper auth/codec validation is left to the worker's camera probe path.
    """
    try:
        parsed = urlsplit(rtsp_url)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            return ProbeResult(ok=False, error_class="decode")
        port = parsed.port or 554
    except ValueError:
        return ProbeResult(ok=False, error_class="decode")
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_sec):
            return ProbeResult(ok=True)
    except TimeoutError:
        return ProbeResult(ok=False, error_class="timeout")
    except OSError:
        return ProbeResult(ok=False, error_class="decode")


def status_from_probe(result: ProbeResult) -> CameraStatus:
    return "online" if result.ok else "offline"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _status(value: object) -> CameraStatus:
    return value if value in {"online", "offline", "starting", "unknown"} else "unknown"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "API_CAMERA_STORE_ENV",
    "DEFAULT_CAMERA_STORE",
    "CameraRegistryStore",
    "CameraStatus",
    "ProbeResult",
    "mask_rtsp_url",
    "probe_rtsp_url",
    "public_camera",
    "status_from_probe",
]
