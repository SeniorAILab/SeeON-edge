from __future__ import annotations

from collections.abc import Mapping

from contracts.worker_config import PulledWorkerConfig
from edge.domains import DOMAIN_REGISTRY
from edge.domains.base import DomainDetector
from edge.runtime.edge_worker_config import CameraRuntimeConfig, EdgeWorkerConfig


def resolve_effective_config(
    yaml_config: EdgeWorkerConfig,
    pulled: PulledWorkerConfig | None,
    *,
    source: str | None = None,
) -> tuple[EdgeWorkerConfig, int, int, str]:
    if pulled is None:
        return yaml_config, 0, 0, "yaml"

    pulled_by_camera_id = {camera.camera_id: camera for camera in pulled.cameras}
    cameras = tuple(
        _override_camera_rtsp(camera, pulled_by_camera_id.get(camera.camera_id).rtsp_url)
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
    yaml_config: EdgeWorkerConfig,
    pulled: PulledWorkerConfig | None,
) -> Mapping[str, object | None]:
    return {
        name: registration.runtime_config_resolver(
            yaml_config.domains.domain_config(name), pulled
        )
        for name, registration in DOMAIN_REGISTRY.items()
        if registration.runtime_config_resolver is not None
    }




def apply_runtime_config(
    detectors: tuple[DomainDetector, ...],
    yaml_config: EdgeWorkerConfig,
    pulled: PulledWorkerConfig | None,
) -> None:
    resolved = resolve_runtime_config(yaml_config, pulled)
    for detector in detectors:
        registration = getattr(detector, "registration", None)
        if registration is not None and registration.runtime_config_updater is not None:
            registration.runtime_config_updater(detector, resolved.get(registration.name))
        elif registration is None:
            for candidate in DOMAIN_REGISTRY.values():
                if candidate.runtime_config_updater is not None:
                    candidate.runtime_config_updater(detector, resolved.get(candidate.name))




def _override_camera_rtsp(camera: CameraRuntimeConfig, rtsp_url: str | None) -> CameraRuntimeConfig:
    if rtsp_url is None:
        return camera
    if camera.streams is not None:
        return camera.model_copy(
            update={"streams": camera.streams.model_copy(update={"sub": rtsp_url})}
        )
    return camera.model_copy(update={"rtsp_url": rtsp_url})


__all__ = ["apply_runtime_config", "resolve_effective_config", "resolve_runtime_config"]
