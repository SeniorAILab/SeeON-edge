from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.worker_config import PulledCameraConfig, PulledNightWindow, PulledWorkerConfig
from worker.runtime.config.camera_models import CameraRuntimeConfig, RelayConfig
from worker.runtime.config.domain_models import DomainsConfig
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError
from worker.runtime.config.restart import RestartDirective
from worker.runtime.config.worker_models import WorkerConfig


class _NightWindowPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    tz: str = Field(min_length=1)


class _CameraPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    camera_id: str = Field(min_length=1)
    facility_id: str | None = Field(default=None, min_length=1)
    space_id: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    rtsp_url: str | None = None
    online: bool = True
    space_name: str | None = None
    floor_name: str | None = None
    created_at: str | None = None
    fps: float | None = Field(default=None, gt=0)
    decode_backend: str | None = None
    domains: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_location(self) -> _CameraPayload:
        if self.facility_id is None and self.space_id is None:
            raise ConfigValidationError("camera must include facility_id or space_id")
        return self

    @property
    def resolved_facility_id(self) -> str:
        return self.facility_id or self.space_id or ""

    @property
    def resolved_space_id(self) -> str:
        return self.space_id or self.facility_id or ""


class BackendWorkerConfigPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    registry_version: int | None = Field(default=None, ge=0)
    config_version: int | None = Field(default=None, ge=0)
    restart_epoch: int | None = Field(default=None, ge=0)
    night_window: _NightWindowPayload | None = None
    cameras: tuple[_CameraPayload, ...]

    @model_validator(mode="after")
    def _require_version(self) -> BackendWorkerConfigPayload:
        if self.registry_version is None and self.config_version is None:
            raise ConfigValidationError("worker config payload must include a version")
        return self

    @property
    def resolved_registry_version(self) -> int:
        return self.registry_version if self.registry_version is not None else 0

    @property
    def directive(self) -> RestartDirective:
        version = self.config_version
        if version is None:
            version = self.resolved_registry_version
        return RestartDirective(
            generation=self.restart_epoch or 0,
            version=version,
        )

    def to_pulled_config(self) -> PulledWorkerConfig:
        window = self.night_window
        return PulledWorkerConfig(
            config_version=self.directive.version,
            restart_epoch=self.directive.generation,
            night_window=(
                None
                if window is None
                else PulledNightWindow(start=window.start, end=window.end, tz=window.tz)
            ),
            cameras=tuple(
                PulledCameraConfig(
                    camera_id=camera.camera_id,
                    space_id=camera.resolved_space_id,
                    label=camera.label or camera.camera_id,
                    rtsp_url=camera.rtsp_url,
                    online=camera.online,
                    space_name=camera.space_name,
                    floor_name=camera.floor_name,
                    created_at=camera.created_at,
                )
                for camera in self.cameras
            ),
        )

    def to_worker_config(self, relay_url: str, relay_token: str | None) -> WorkerConfig:
        token = "" if relay_token is None else relay_token.strip()
        if not token:
            raise WorkerConfigError("RELAY_TOKEN is required for pulled worker config")
        cameras = tuple(
            _runtime_camera(camera)
            for camera in self.cameras
            if camera.rtsp_url is not None
        )
        if not cameras:
            raise WorkerConfigError("worker config must include at least one camera")
        domains = tuple(sorted({name for camera in self.cameras for name in camera.domains}))
        return WorkerConfig(
            relay=RelayConfig.model_validate({"url": relay_url, "token": token}),
            domains=DomainsConfig(enabled=domains or None),
            cameras=cameras,
        )


def _runtime_camera(payload: _CameraPayload) -> CameraRuntimeConfig:
    if payload.rtsp_url is None:
        raise WorkerConfigError("worker camera is missing an RTSP URL")
    return CameraRuntimeConfig(
        camera_id=payload.camera_id,
        facility_id=payload.resolved_facility_id,
        rtsp_url=payload.rtsp_url,
        fps=payload.fps or 5.0,
        decode_backend=payload.decode_backend,
        label=payload.label,
    )


__all__ = ["BackendWorkerConfigPayload"]
