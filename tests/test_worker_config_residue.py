"""Residue from `tests/test_edge_worker_config.py` (now deleted).

That file tested `edge.runtime.edge_worker_config`'s YAML-load and
`CameraRuntimeConfig` model-validation contract (streams, decode backends,
fps, duplicate camera ids, relay URL shape, resident_id normalization, and
secret redaction in errors) -- NOT the config pull/LKG/restart-directive
lifecycle its filename suggested; that lifecycle behavior lives in
`edge.runtime.config_pull` and is what
`tests/test_worker_config_pull_lkg.py`, `tests/test_worker_config_lifecycle.py`,
and `tests/test_worker_restart_directive.py` actually migrated.

Of the 16 tests in the legacy file, one (`test_edge_worker_config_rejects_backend_credentials`)
is verifiably superseded by
`tests/test_worker_backend_ingest_contract.py::test_worker_config_rejects_camera_backend_credentials`
(tests/test_worker_backend_ingest_contract.py:51-59), which parametrizes over
both `ingest_key_id` and `ingest_secret` against `WorkerConfig.model_validate`
directly and is a strict superset. The other 15 have no successor anywhere in
the currently-migrated worker test suite and are ported below onto
`worker.runtime.config` (`CameraRuntimeConfig`, `WorkerConfig`,
`WorkerConfigError`, `load_worker_config`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

from worker.runtime.config import (
    CameraRuntimeConfig,
    WorkerConfigError,
    load_worker_config,
)


def test_worker_config_loads_four_cameras_and_redacts_relay_token(tmp_path: Path) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(edge_config_payload()), encoding="utf-8")

    config = load_worker_config(config_path)

    assert len(config.cameras) == 4
    assert [camera.camera_id for camera in config.cameras] == [
        "camera-1",
        "camera-2",
        "camera-3",
        "camera-4",
    ]
    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert "relay-token-1" not in repr(config)


def test_worker_config_rejects_duplicate_camera_ids(tmp_path: Path) -> None:
    payload = edge_config_payload()
    payload["cameras"][1]["camera_id"] = "camera-1"
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="duplicate camera_id"):
        load_worker_config(config_path)


def test_worker_config_requires_rtsp_url(tmp_path: Path) -> None:
    payload = edge_config_payload()
    payload["cameras"][0]["rtsp_url"] = ""
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="rtsp_url"):
        load_worker_config(config_path)


def test_worker_config_normalizes_blank_resident_id(tmp_path: Path) -> None:
    payload = edge_config_payload()
    payload["cameras"][0]["resident_id"] = "  "
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(config_path)

    assert config.cameras[0].resident_id is None


def test_camera_runtime_config_defaults_to_opencv_decode_backend() -> None:
    config = CameraRuntimeConfig(
        camera_id="camera-1",
        facility_id="facility-1",
        rtsp_url="rtsp://camera.local/trackID=2",
    )

    assert config.decode_backend is None
    assert config.fps == 5.0
    assert config.inference_rtsp_url == "rtsp://camera.local/trackID=2"
    assert config.main_rtsp_url is None


def test_camera_runtime_config_accepts_opencv_decode_backend() -> None:
    config = CameraRuntimeConfig(
        camera_id="camera-1",
        facility_id="facility-1",
        rtsp_url="rtsp://camera.local/trackID=2",
        decode_backend="OpenCV",
    )

    assert config.decode_backend == "opencv"


def test_camera_runtime_config_accepts_nvdec_and_auto_decode_backend() -> None:
    for value, expected in (("NVDEC", "nvdec"), ("Auto", "auto"), ("cpu", "cpu")):
        config = CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            rtsp_url="rtsp://camera.local/trackID=2",
            decode_backend=value,
        )
        assert config.decode_backend == expected


def test_camera_runtime_config_accepts_dual_streams_and_fps() -> None:
    config = CameraRuntimeConfig(
        camera_id="camera-1",
        facility_id="facility-1",
        streams={
            "sub": " rtsp://camera.local/sub ",
            "main": "rtsp://camera.local/main",
        },
        fps=7.5,
    )

    assert config.rtsp_url is None
    assert config.inference_rtsp_url == "rtsp://camera.local/sub"
    assert config.main_rtsp_url == "rtsp://camera.local/main"
    assert config.fps == 7.5


def test_camera_runtime_config_prefers_sub_stream_for_inference() -> None:
    config = CameraRuntimeConfig(
        camera_id="camera-1",
        facility_id="facility-1",
        rtsp_url="rtsp://camera.local/legacy",
        streams={"sub": "rtsp://camera.local/sub"},
    )

    assert config.inference_rtsp_url == "rtsp://camera.local/sub"


def test_camera_runtime_config_rejects_missing_inference_url() -> None:
    with pytest.raises(ValueError, match="rtsp_url or streams.sub"):
        CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
        )


def test_camera_runtime_config_rejects_invalid_stream_url() -> None:
    with pytest.raises(ValueError, match="streams must start with rtsp://"):
        CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            streams={"sub": "http://camera.local/sub"},
        )


def test_camera_runtime_config_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            rtsp_url="rtsp://camera.local/trackID=2",
            fps=0,
        )


def test_camera_runtime_config_rejects_unknown_decode_backend() -> None:
    with pytest.raises(ValueError, match="decode_backend must be one of"):
        CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            rtsp_url="rtsp://camera.local/trackID=2",
            decode_backend="gstreamer",
        )


def test_worker_config_rejects_relative_relay_url(tmp_path: Path) -> None:
    payload = edge_config_payload()
    payload["relay"]["url"] = "/relay"
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="relay.url"):
        load_worker_config(config_path)


def test_worker_config_error_does_not_include_relay_token_value(tmp_path: Path) -> None:
    payload = edge_config_payload()
    payload["relay"]["token"] = "super-secret-value"
    payload["cameras"][0]["rtsp_url"] = "not-rtsp"
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError) as exc_info:
        load_worker_config(config_path)

    assert "super-secret-value" not in str(exc_info.value)
