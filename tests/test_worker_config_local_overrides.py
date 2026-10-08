from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Self, final

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.runtime.config import (
    ClipRecordingConfig,
    ConfigSource,
    JsonObject,
    WorkerConfig,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    resolve_local_overrides,
)
from worker.runtime.config.local_env import ML_WORKER_CLIP_RECORDING_ENABLED_ENV


@final
class FakeResponse:
    def __init__(self, payload: JsonObject, status: int = 200) -> None:
        self._payload: JsonObject = payload
        self.status: int = status

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _write_fall_artifact(path: Path) -> Path:
    return write_pose_bbox56_bundle(path)


def _payload(*, registry_version: int, config_version: int, restart_epoch: int) -> JsonObject:
    return {
        "registry_version": registry_version,
        "config_version": config_version,
        "restart_epoch": restart_epoch,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://user:camera-pass@camera/live",
                "fps": 7.5,
                "domains": ["fall"],
            }
        ],
    }


def test_pull_with_clip_env_set_configures_packaged_fall_and_clip_enabled(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    artifact_dir = packaged_fall_bundle
    environ = {ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip, dev_mjpeg = resolve_local_overrides(None, environ)

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is not None
    assert snapshot.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert snapshot.config.clip.enabled is True


def test_pull_with_no_fall_config_resolves_the_packaged_default_bundle(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    models, clip, dev_mjpeg = resolve_local_overrides(None, {})
    assert models.fall is not None
    assert models.fall.type == "pose-bbox56-proxy-v0"
    assert models.fall.artifact_dir == packaged_fall_bundle.resolve()
    assert clip.enabled is ClipRecordingConfig().enabled
    assert dev_mjpeg is None

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is not None
    assert snapshot.config.models.fall.type == "pose-bbox56-proxy-v0"
    assert snapshot.config.clip.enabled is ClipRecordingConfig().enabled
    assert snapshot.config.dev_mjpeg.enabled is True
    assert snapshot.config.dev_mjpeg.host == "0.0.0.0"
    assert snapshot.config.dev_mjpeg.port == 8090


def test_local_yaml_fall_config_is_kept_when_clip_env_is_set(tmp_path: Path) -> None:
    yaml_artifact_dir = _write_fall_artifact(tmp_path / "yaml-fall")
    yaml_config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://ml-api:8000", "token": "relay-secret"},
            "cameras": [
                {
                    "camera_id": "yaml-camera",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://yaml/camera",
                }
            ],
            "models": {
                "fall": {
                    "type": "pose-bbox56-proxy-v0",
                    "framework": "pytorch",
                    "mode": "sequence",
                    "artifact_dir": str(yaml_artifact_dir),
                    "window": 30,
                    "stride": 5,
                    "input_shape": [30, 56],
                    "operating_threshold": 0.5,
                }
            },
            "clip": {"enabled": True},
        }
    )
    environ = {ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "false"}

    models, clip, dev_mjpeg = resolve_local_overrides(yaml_config, environ)

    assert models.fall is not None
    assert models.fall.artifact_dir == yaml_artifact_dir.resolve()
    assert clip.enabled is True
    assert dev_mjpeg is None


def test_lkg_restore_path_preserves_locally_sourced_models_and_clip(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    artifact_dir = packaged_fall_bundle
    environ = {ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip, dev_mjpeg = resolve_local_overrides(None, environ)
    store = WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3")

    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=3, config_version=3, restart_epoch=1)
        ),
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )
    assert fresh is not None
    assert fresh.source is ConfigSource.PULLED

    def offline(_request: urllib.request.Request, _timeout: float) -> FakeResponse:
        raise urllib.error.URLError("offline")

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=offline,
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.stale is True
    assert stale.config.models.fall is not None
    assert stale.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert stale.config.clip.enabled is True


def test_pull_with_yaml_dev_mjpeg_enabled_survives_the_pull(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    yaml_config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://ml-api:8000", "token": "relay-secret"},
            "cameras": [
                {
                    "camera_id": "yaml-camera",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://yaml/camera",
                }
            ],
            "dev_mjpeg": {"enabled": True, "host": "127.0.0.1", "port": 8090},
        }
    )
    _models, _clip, dev_mjpeg = resolve_local_overrides(yaml_config, {})
    assert dev_mjpeg is not None
    assert dev_mjpeg.enabled is True

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=_models,
        clip=_clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.dev_mjpeg.enabled is True
    assert snapshot.config.dev_mjpeg.port == 8090


def test_pull_with_no_yaml_dev_mjpeg_leaves_it_disabled_for_env_fallback(
    packaged_fall_bundle: Path,
) -> None:
    _models, _clip, dev_mjpeg = resolve_local_overrides(None, {})
    assert dev_mjpeg is None
