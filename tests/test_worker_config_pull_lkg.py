from __future__ import annotations

import urllib.error
import urllib.request

from contracts.worker_config import PulledCameraConfig, PulledWorkerConfig
from worker.runtime.config import (
    ML_WORKER_STATE_DIR_ENV,
    CameraRuntimeConfig,
    CameraStreamsConfig,
    RelayConfig,
    WorkerConfig,
    load_lkg,
    pull_worker_config,
    resolve_effective_config,
    save_lkg,
)


def test_lkg_save_load_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ML_WORKER_STATE_DIR_ENV, str(tmp_path))
    cfg = _pulled(config_version=7, restart_epoch=3, rtsp_url="rtsp://pulled/sub")

    save_lkg(cfg)

    assert load_lkg() == cfg


def test_pull_worker_config_returns_none_on_urllib_error() -> None:
    # The worker transport (http_transport.stdlib_urlopen) is http.client-based,
    # not urllib.request.urlopen, so the fake transport is injected via the
    # `urlopen` parameter rather than monkeypatched globally.
    def _raise(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("offline")

    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise)
        is None
    )


def test_resolve_effective_config_pull_ok_overrides_rtsp_and_versions() -> None:
    yaml_config = _yaml_config()
    pulled = _pulled(config_version=11, restart_epoch=5, rtsp_url="rtsp://pulled/sub")

    effective, config_version, restart_epoch, source = resolve_effective_config(
        yaml_config,
        pulled,
    )

    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://pulled/sub"
    assert effective.cameras[1].rtsp_url == "rtsp://yaml/legacy"
    assert config_version == 11
    assert restart_epoch == 5
    assert source == "pulled"


def test_resolve_effective_config_pull_fail_lkg_present_uses_lkg_values() -> None:
    yaml_config = _yaml_config()
    lkg = _pulled(config_version=13, restart_epoch=8, rtsp_url="rtsp://lkg/sub")

    effective, config_version, restart_epoch, source = resolve_effective_config(
        yaml_config,
        lkg,
        source="lkg",
    )

    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://lkg/sub"
    assert config_version == 13
    assert restart_epoch == 8
    assert source == "lkg"


def test_resolve_effective_config_pull_fail_no_lkg_uses_yaml_unchanged() -> None:
    yaml_config = _yaml_config()

    effective, config_version, restart_epoch, source = resolve_effective_config(yaml_config, None)

    assert effective is yaml_config
    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://yaml/sub"
    assert config_version == 0
    assert restart_epoch == 0
    assert source == "yaml"


def _yaml_config() -> WorkerConfig:
    return WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="token"),
        cameras=(
            CameraRuntimeConfig(
                camera_id="camera-1",
                facility_id="facility-1",
                streams=CameraStreamsConfig(sub="rtsp://yaml/sub", main="rtsp://yaml/main"),
            ),
            CameraRuntimeConfig(
                camera_id="camera-2",
                facility_id="facility-1",
                rtsp_url="rtsp://yaml/legacy",
            ),
        ),
    )


def _pulled(config_version: int, restart_epoch: int, rtsp_url: str) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=config_version,
        restart_epoch=restart_epoch,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="camera-1",
                space_id="space-1",
                label="Room 1",
                rtsp_url=rtsp_url,
                online=True,
            ),
        ),
    )


def test_unavailable_pull_returns_none_and_preserves_existing_lkg(
    tmp_path,
    monkeypatch,
) -> None:
    # Regression: when ml-api has no backend config it returns 503, so the pull
    # MUST return None and the worker MUST keep its existing LKG (not overwrite
    # it with an empty placeholder). config_pull.load_worker_config_from_relay /
    # pull_worker_config only persist the LKG on a successful, validated pull.
    monkeypatch.setenv(ML_WORKER_STATE_DIR_ENV, str(tmp_path))
    good = _pulled(config_version=5, restart_epoch=1, rtsp_url="rtsp://lkg/good")
    save_lkg(good)

    def _raise_503(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(
            "http://ml-api:8000/api/v1/cameras/worker-config", 503, "unavailable", {}, None
        )

    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise_503)
        is None
    )
    # Existing LKG is intact: an unavailable pull never clobbers last-known-good.
    assert load_lkg() == good
