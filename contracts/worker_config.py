"""Worker runtime-config pull contract (ml-api camera registry SSOT -> worker).

Dependency-light shared shape for the config the worker pulls from ml-api.
The worker roster is authoritative at ``/api/v1/cameras/worker-config`` and is
derived from the ml-api camera registry; ``/api/v1/relay/config`` is only a
backward-compatible alias. Backend-pulled ML settings such as night window and
config version are optional metadata on that same response, so the worker does
not consume a second config authority.
"""

from __future__ import annotations

from dataclasses import dataclass

# Worker-facing paths (full suffixes mounted under the /api/v1 app prefix).
WORKER_CONFIG_PATH = "/api/v1/cameras/worker-config"
WORKER_RESTART_PATH = "/api/v1/relay/restart"

CONFIG_VERSION_KEY = "config_version"
RESTART_EPOCH_KEY = "restart_epoch"


@dataclass(frozen=True, slots=True)
class PulledCameraConfig:
    camera_id: str
    space_id: str
    label: str
    rtsp_url: str | None
    online: bool
    space_name: str | None = None
    floor_name: str | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledCameraConfig:
        return cls(
            camera_id=_require_str(data, "camera_id"),
            space_id=_require_str(data, "space_id"),
            label=_require_str(data, "label"),
            rtsp_url=_optional_str(data, "rtsp_url"),
            online=bool(data.get("online", False)),
            space_name=_optional_str(data, "space_name"),
            floor_name=_optional_str(data, "floor_name"),
            created_at=_optional_str(data, "created_at"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "space_id": self.space_id,
            "label": self.label,
            "rtsp_url": self.rtsp_url,
            "online": self.online,
            "space_name": self.space_name,
            "floor_name": self.floor_name,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PulledNightWindow:
    start: str
    end: str
    tz: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledNightWindow:
        return cls(
            start=_require_str(data, "start"),
            end=_require_str(data, "end"),
            tz=_require_str(data, "tz"),
        )

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "tz": self.tz}


@dataclass(frozen=True, slots=True)
class PulledWorkerConfig:
    config_version: int
    restart_epoch: int
    night_window: PulledNightWindow | None
    cameras: tuple[PulledCameraConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledWorkerConfig:
        raw_window = data.get("night_window")
        raw_cameras = data.get("cameras") or []
        if not isinstance(raw_cameras, list):
            raise TypeError("cameras must be a list")
        return cls(
            config_version=_require_int(data, "config_version"),
            restart_epoch=_require_int(data, "restart_epoch"),
            night_window=(
                PulledNightWindow.from_dict(raw_window)
                if isinstance(raw_window, dict)
                else None
            ),
            cameras=tuple(
                PulledCameraConfig.from_dict(item)
                for item in raw_cameras
                if isinstance(item, dict)
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "restart_epoch": self.restart_epoch,
            "night_window": (
                None if self.night_window is None else self.night_window.as_dict()
            ),
            "cameras": [camera.as_dict() for camera in self.cameras],
        }


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    # bool is an int subclass; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


__all__ = [
    "WORKER_CONFIG_PATH",
    "WORKER_RESTART_PATH",
    "CONFIG_VERSION_KEY",
    "RESTART_EPOCH_KEY",
    "PulledCameraConfig",
    "PulledNightWindow",
    "PulledWorkerConfig",
]
