from __future__ import annotations

from collections.abc import Mapping

from contracts.worker_config import PulledWorkerConfig
from worker.runtime.config.camera_models import CameraRuntimeConfig
from worker.runtime.config.domain_models import NightWindowConfig
from worker.runtime.config.worker_models import WorkerConfig


def resolve_effective_config(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
    *,
    source: str | None = None,
) -> tuple[WorkerConfig, int, int, str]:
    if pulled is None:
        return yaml_config, 0, 0, "yaml"
    pulled_by_camera_id = {camera.camera_id: camera for camera in pulled.cameras}
    cameras = tuple(
        _override_camera_rtsp(
            camera,
            pulled_by_camera_id[camera.camera_id].rtsp_url,
        )
        if camera.camera_id in pulled_by_camera_id
        else camera
        for camera in yaml_config.cameras
    )
    return (
        yaml_config.model_copy(update={"cameras": cameras}),
        pulled.config_version,
        pulled.restart_epoch,
        "pulled" if source is None else source,
    )


def resolve_runtime_config(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
) -> Mapping[str, NightWindowConfig | None]:
    pulled_window = None if pulled is None else pulled.night_window
    if pulled_window is not None:
        window = NightWindowConfig(
            start=pulled_window.start,
            end=pulled_window.end,
            tz=pulled_window.tz,
        )
    else:
        bed_exit = yaml_config.domains.bed_exit
        window = None if bed_exit is None else bed_exit.night_window
    return {"bed_exit": window}


def _override_camera_rtsp(
    camera: CameraRuntimeConfig,
    rtsp_url: str | None,
) -> CameraRuntimeConfig:
    if rtsp_url is None:
        return camera
    if camera.streams is not None:
        return camera.model_copy(
            update={"streams": camera.streams.model_copy(update={"sub": rtsp_url})}
        )
    return camera.model_copy(update={"rtsp_url": rtsp_url})


__all__ = ["resolve_effective_config", "resolve_runtime_config"]
