from __future__ import annotations

import sys
from pathlib import PurePosixPath
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from contracts.worker_config import (
    PulledCameraConfig,
    PulledNightWindow,
    PulledWorkerConfig,
    detection_window_validation_error,
)
from shared.detection_policies import (
    PolicyBundle,
    PolicyDocumentError,
    default_policy_bundle,
    parse_policy_bundle,
)
from worker.runtime.config.camera_models import (
    BedZoneRegionConfig,
    CameraRuntimeConfig,
    RelayConfig,
)
from worker.runtime.config.domain_models import (
    KNOWN_DOMAIN_NAMES,
    BedExitDomainConfig,
    DomainsConfig,
    FallDomainConfig,
    NightWindowConfig,
)
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError
from worker.runtime.config.restart import RestartDirective
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    WorkerConfig,
    WorkerModelsConfig,
)
from worker.types import CURRENT_TEMPORAL_PROFILE


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
    frame_stride: int | None = Field(default=None, gt=0)
    decode_backend: str | None = None
    domains: tuple[str, ...] | None = None
    bed_zone_regions: tuple[BedZoneRegionConfig, ...] = Field(default=(), max_length=8)
    bed_zone_image_width: int | None = Field(default=None, gt=0)
    bed_zone_image_height: int | None = Field(default=None, gt=0)

    @property
    def resolved_facility_id(self) -> str:
        if self.facility_id is not None:
            return self.facility_id
        return "local"

    @property
    def resolved_space_id(self) -> str:
        return self.space_id or self.facility_id or ""


class BackendWorkerConfigPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    registry_version: int | None = Field(default=None, ge=0)
    config_version: int | None = Field(default=None, ge=0)
    restart_epoch: int | None = Field(default=None, ge=0)
    night_window: _NightWindowPayload | None = None
    detection_windows: dict[str, object] | None = None
    cameras: tuple[object, ...]
    domains: dict[str, object] | None = None
    clip_store_subdir: object = None
    detection_policies: object = None
    clip_export_enabled: StrictBool = False
    clip_export_version: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _reject_unimplemented_policy_payload(cls, data: object) -> object:
        if isinstance(data, dict):
            fields = sorted({"models", "clip"}.intersection(data))
            if fields:
                raise ConfigValidationError(
                    "worker config policy field(s) not accepted before Todo 9: " + ", ".join(fields)
                )
        return data

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
            registry=self.resolved_registry_version,
        )

    @property
    def resolved_detection_windows(self) -> dict[str, PulledNightWindow]:
        if self.detection_windows is not None:
            windows: dict[str, PulledNightWindow] = {}
            for domain, window in self.detection_windows.items():
                if window is None:
                    continue
                if not isinstance(window, dict):
                    _log_invalid_detection_window(domain, window, "must be an object or null")
                    continue
                try:
                    payload = _NightWindowPayload.model_validate(window)
                except ValidationError as exc:
                    _log_invalid_detection_window(domain, window, _validation_error_reason(exc))
                    continue
                validated = _validated_pulled_window(domain, payload)
                if validated is not None:
                    windows[domain] = validated
            return windows
        window = self.night_window
        if window is None:
            return {}
        validated = _validated_pulled_window("bed_exit", window)
        return {} if validated is None else {"bed_exit": validated}

    @property
    def resolved_domain_enabled(self) -> dict[str, bool]:
        if self.domains is None:
            return {}
        resolved: dict[str, bool] = {}
        for domain, value in self.domains.items():
            if not isinstance(domain, str) or domain not in KNOWN_DOMAIN_NAMES:
                _log_invalid_domain_config(domain, value, "unknown or non-string domain")
                continue
            if not isinstance(value, dict):
                _log_invalid_domain_config(domain, value, "must be an object")
                continue
            enabled = value.get("enabled")
            if not isinstance(enabled, bool):
                _log_invalid_domain_config(domain, value, "enabled must be a boolean")
                continue
            resolved[domain] = enabled
        return resolved

    @property
    def resolved_clip_store_subdir(self) -> str | None:
        value = self.clip_store_subdir
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            _log_invalid_clip_store_subdir(value, "must be a non-empty string")
            return None
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            _log_invalid_clip_store_subdir(value, "must be a relative path without .. segments")
            return None
        return value

    @property
    def resolved_cameras(self) -> tuple[_CameraPayload, ...]:
        cameras: list[_CameraPayload] = []
        for entry in self.cameras:
            if not isinstance(entry, dict):
                _log_invalid_camera(entry, "must be an object")
                continue
            try:
                camera = _CameraPayload.model_validate(entry)
            except ValidationError as exc:
                _log_invalid_camera(entry, _validation_error_reason(exc))
                continue
            cameras.append(camera)
        return tuple(cameras)

    @property
    def resolved_detection_policies(self) -> PolicyBundle:
        if self.detection_policies is None:
            return default_policy_bundle(
                tuple(camera.camera_id for camera in self.resolved_cameras)
            )
        try:
            return parse_policy_bundle(self.detection_policies)
        except PolicyDocumentError as error:
            raise WorkerConfigError(f"detection policy refused: {error}") from error

    def to_pulled_config(self) -> PulledWorkerConfig:
        detection_windows = self.resolved_detection_windows
        _ = self.resolved_detection_policies
        return PulledWorkerConfig(
            config_version=self.directive.version,
            restart_epoch=self.directive.generation,
            night_window=detection_windows.get("bed_exit"),
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
                for camera in self.resolved_cameras
            ),
            registry_version=self.directive.registry,
            detection_windows=detection_windows,
        )

    def to_worker_config(
        self,
        relay_url: str,
        relay_token: str | None,
        *,
        models: WorkerModelsConfig | None = None,
        clip: ClipRecordingConfig | None = None,
        dev_mjpeg: DevMjpegConfig | None = None,
    ) -> WorkerConfig:
        token = "" if relay_token is None else relay_token.strip()
        if not token:
            raise WorkerConfigError("RELAY_TOKEN is required for pulled worker config")
        resolved_cameras = self.resolved_cameras
        cameras = tuple(
            _runtime_camera(camera) for camera in resolved_cameras if camera.rtsp_url is not None
        )
        if self.cameras and not resolved_cameras:
            raise WorkerConfigError("worker config declared cameras but none of them parsed")
        detection_windows: dict[str, NightWindowConfig | None] = {
            domain: NightWindowConfig(start=window.start, end=window.end, tz=window.tz)
            for domain, window in self.resolved_detection_windows.items()
        }
        domain_enabled = self.resolved_domain_enabled
        if self.domains is not None:
            domains_config = DomainsConfig(
                fall=(
                    FallDomainConfig(enabled=domain_enabled["fall"])
                    if "fall" in domain_enabled
                    else None
                ),
                bed_exit=(
                    BedExitDomainConfig(enabled=domain_enabled["bed_exit"])
                    if "bed_exit" in domain_enabled
                    else None
                ),
                detection_windows=detection_windows or None,
            )
        else:
            camera_declared_domains = any(camera.domains is not None for camera in resolved_cameras)
            camera_domains = (
                tuple(
                    sorted({name for camera in resolved_cameras for name in (camera.domains or ())})
                )
                if camera_declared_domains
                else None
            )
            domains_config = DomainsConfig(
                enabled=camera_domains,
                detection_windows=detection_windows or None,
            )
        base_clip = clip if clip is not None else ClipRecordingConfig()
        subdir = self.resolved_clip_store_subdir
        resolved_clip = (
            base_clip if subdir is None else base_clip.model_copy(update={"store_subdir": subdir})
        )
        return WorkerConfig(
            version=self.directive.version,
            relay=RelayConfig.model_validate({"url": relay_url, "token": token}),
            models=models,
            domains=domains_config,
            detection_policies=self.resolved_detection_policies,
            clip=resolved_clip,
            dev_mjpeg=(
                dev_mjpeg
                if dev_mjpeg is not None
                else DevMjpegConfig(enabled=True, host="0.0.0.0", port=8090)
            ),
            clip_export_enabled=self.clip_export_enabled,
            clip_export_version=self.clip_export_version,
            cameras=cameras,
        )


def _validated_pulled_window(domain: str, window: _NightWindowPayload) -> PulledNightWindow | None:
    reason = detection_window_validation_error(window.start, window.end, window.tz)
    if reason is not None:
        print(
            f"detection window for domain {domain!r} is invalid ({reason}): "
            f"start={window.start!r} end={window.end!r} tz={window.tz!r}; "
            "falling open to ALWAYS/24-7 detection for this domain",
            file=sys.stderr,
        )
        return None
    return PulledNightWindow(start=window.start, end=window.end, tz=window.tz)


def _log_invalid_detection_window(domain: str, value: object, reason: str) -> None:
    print(
        f"detection window for domain {domain!r} is invalid ({reason}): {value!r}; "
        "falling open to ALWAYS/24-7 detection for this domain",
        file=sys.stderr,
    )


def _log_invalid_domain_config(domain: object, value: object, reason: str) -> None:
    print(
        f"domain enable/disable override for domain {domain!r} is invalid ({reason}): "
        f"{value!r}; ignoring this domain's override",
        file=sys.stderr,
    )


def _log_invalid_clip_store_subdir(value: object, reason: str) -> None:
    print(
        f"clip_store_subdir is invalid ({reason}): {value!r}; falling back to the clip store root",
        file=sys.stderr,
    )


def _log_invalid_camera(value: object, reason: str) -> None:
    identifier = value.get("camera_id") if isinstance(value, dict) else None
    label = f"camera_id={identifier!r}" if identifier else f"entry={value!r}"
    print(
        f"camera config entry is invalid ({reason}): {label}; dropping this camera",
        file=sys.stderr,
    )


def _validation_error_reason(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _runtime_camera(payload: _CameraPayload) -> CameraRuntimeConfig:
    if payload.rtsp_url is None:
        raise WorkerConfigError("worker camera is missing an RTSP URL")
    return CameraRuntimeConfig(
        camera_id=payload.camera_id,
        facility_id=payload.resolved_facility_id,
        rtsp_url=payload.rtsp_url,
        fps=payload.fps or CURRENT_TEMPORAL_PROFILE.target_fps,
        frame_stride=payload.frame_stride or 1,
        decode_backend=payload.decode_backend,
        label=payload.label,
        bed_zone_regions=payload.bed_zone_regions,
        bed_zone_image_width=payload.bed_zone_image_width,
        bed_zone_image_height=payload.bed_zone_image_height,
    )


__all__ = ["BackendWorkerConfigPayload"]
