from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from worker.runtime.config import (
    BackendWorkerConfigPayload,
    ConfigSource,
    NightWindowConfig,
    RestartDirective,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    pull_worker_config,
)


def test_to_worker_config_threads_pulled_detection_windows_into_domains_config() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
                "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "bed_exit": NightWindowConfig(start="21:00", end="06:00", tz="UTC"),
        "fall": NightWindowConfig(start="22:00", end="05:00", tz="Asia/Seoul"),
    }
    assert worker_config.domains.resolved_detection_window("bed_exit") == NightWindowConfig(
        start="21:00", end="06:00", tz="UTC"
    )


def test_to_worker_config_threads_bed_zone_regions_into_camera_runtime_config() -> None:
    regions = [
        {
            "id": f"bed-{index}",
            "polygon": [
                [index * 20 + 1, 2],
                [index * 20 + 9, 2],
                [index * 20 + 9, 8],
                [index * 20 + 1, 8],
            ],
            "origin": "manual" if index % 2 == 0 else "model",
        }
        for index in range(1, 5)
    ]
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                    "bed_zone_regions": regions,
                    "bed_zone_image_width": 640,
                    "bed_zone_image_height": 480,
                },
                {
                    "camera_id": "camera-2",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-2/stream",
                },
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    cameras = {camera.camera_id: camera for camera in worker_config.cameras}
    assert [
        region.model_dump(mode="json") for region in cameras["camera-1"].bed_zone_regions
    ] == regions
    assert cameras["camera-1"].bed_zone_image_width == 640
    assert cameras["camera-1"].bed_zone_image_height == 480
    assert cameras["camera-2"].bed_zone_regions == ()
    assert cameras["camera-2"].bed_zone_image_width is None
    assert cameras["camera-2"].bed_zone_image_height is None


def test_to_worker_config_with_empty_camera_list_boots_with_an_empty_roster() -> None:
    payload = BackendWorkerConfigPayload.model_validate({"config_version": 1, "cameras": []})

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.cameras == ()


def test_to_worker_config_with_every_camera_missing_rtsp_url_boots_with_an_empty_roster() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [
                {"camera_id": "camera-1", "facility_id": "facility-1", "rtsp_url": None},
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.cameras == ()


def test_to_worker_config_still_accepts_legacy_night_window_payload_field() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "bed_exit": NightWindowConfig(start="21:00", end="06:00", tz="UTC"),
    }


def test_to_worker_config_drops_start_equal_end_window_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "09:00", "end": "09:00", "tz": "UTC"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows is None
    assert worker_config.domains.resolved_detection_window("bed_exit") is None
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_to_worker_config_drops_invalid_timezone_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "fall": {"start": "22:00", "end": "05:00", "tz": "Not/A_Zone"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows is None
    err = capsys.readouterr().err
    assert "fall" in err


def test_to_pulled_config_drops_malformed_hhmm_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "25:00", "end": "17:00", "tz": "UTC"},
            },
        }
    )

    pulled = payload.to_pulled_config()

    assert pulled.detection_windows == {}
    assert pulled.night_window is None
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_to_worker_config_drops_explicit_null_domain_entry_without_crashing_payload() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": None,
                "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "fall": NightWindowConfig(start="22:00", end="05:00", tz="UTC"),
    }
    assert worker_config.domains.resolved_detection_window("bed_exit") is None


def test_pull_worker_config_returns_none_on_urllib_error() -> None:
    def _raise(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("offline")

    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise) is None
    )


def test_load_on_fresh_central_edge_db_returns_none_not_migration_error(tmp_path) -> None:
    store = WorkerConfigLkgStore(tmp_path / "edge.sqlite3")
    assert not (tmp_path / "edge.sqlite3").exists()

    assert store.load() is None
    assert not (tmp_path / "edge.sqlite3").exists()


def test_unavailable_pull_returns_none_and_preserves_existing_lkg(tmp_path) -> None:
    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise_503)
        is None
    )

    store = WorkerConfigLkgStore(tmp_path / "worker-config.sqlite3")
    good_payload = {
        "registry_version": 5,
        "config_version": 5,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://lkg/good",
            }
        ],
    }

    def _respond_good(request: urllib.request.Request, timeout: float) -> object:
        return _FakeResponse(good_payload)

    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "token",
        store=store,
        urlopen=_respond_good,
    )
    assert fresh is not None
    assert fresh.source is ConfigSource.PULLED
    assert fresh.directive == RestartDirective(generation=1, version=5, registry=5)

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "token",
        store=store,
        urlopen=_raise_503,
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.registry_version == 5
    assert stale.directive == RestartDirective(generation=1, version=5, registry=5)


def _raise_503(request: urllib.request.Request, timeout: float) -> object:
    raise urllib.error.HTTPError(
        "http://ml-api:8000/api/v1/cameras/worker-config", 503, "unavailable", {}, None
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
