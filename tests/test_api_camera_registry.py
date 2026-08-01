from __future__ import annotations

import json
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Self, TypedDict

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.cameras.store import CameraRegistryStore, public_camera
from backend.app.lifespan import refresh_backend_config
from backend.app.main import create_app, no_lifespan
from backend.app.shared.backend_mapping import MappingResult
from contracts.worker_config import PulledCameraConfig, PulledNightWindow, PulledWorkerConfig
from worker.runtime.config import JsonObject, WorkerConfigLkgStore, load_worker_config_from_relay

AUTH = {"Authorization": "Bearer relay-token"}
REPO_ROOT = Path(__file__).resolve().parents[1]


class CapturedBackendCall(TypedDict):
    url: str
    method: str
    authorization: str | None
    facility_id: str | None
    body: dict[str, object]
    timeout: float


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "API_CAMERA_STORE",
        "API_EDGE_RELAY_TOKEN",
        "API_FACILITY_ID",
        "API_BACKEND_EDGE_CAMERAS_URL",
        "API_BACKEND_URL",
        "API_BACKEND_EVENTS_URL",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
        "ML_EDGE_VERSION",
        "ML_WORKER_STATE_DIR",
        "RELAY_TOKEN",
        "RELAY_URL",
        "ML_API_WORKER_PROBE_ORIGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_camera_registry_crud_masks_rtsp_versions_and_worker_config_auth(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_CAMERA_STORE", str(tmp_path / "cameras.json"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setenv("API_BACKEND_EDGE_CAMERAS_URL", "http://backend/api/v1/edge/cameras")
    monkeypatch.setenv("API_FACILITY_TOKEN", "facility-token")
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")
    captured: list[CapturedBackendCall] = []
    probe_calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        if request.full_url == "http://worker.local:8090/probe":
            probe_calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "relay_token": request.headers.get("X-edge-relay-token"),
                    "body": json.loads(request.data.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return FakeHTTPResponse({"ok": False, "error_class": "timeout"})
        decoded_body = json.loads(request.data.decode("utf-8"))
        assert isinstance(decoded_body, dict)
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "authorization": request.headers.get("Authorization"),
                "facility_id": request.headers.get("X-facility-id"),
                "body": {str(key): value for key, value in decoded_body.items()},
                "timeout": timeout,
            }
        )
        return FakeHTTPResponse({"cameraId": "backend-camera-1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://user:secret@camera.local:8554/live",
                "space_id": "space-1",
            },
        )
        assert created.status_code == 201
        camera = created.json()
        assert camera["label"] == "Lobby"
        assert camera["rtsp_url_masked"] == "rtsp://***:***@redacted-camera:8554/live"
        camera_json = json.dumps(camera)
        assert "secret" not in camera_json
        assert "camera.local" not in camera_json
        assert camera["backend_camera_id"] == "backend-camera-1"
        assert camera["id"] == "backend-camera-1"
        assert camera["status"] == "offline"

        listed = client.get("/api/v1/cameras", headers=AUTH).json()
        assert listed["registry_version"] == 1
        assert listed["cameras"] == [camera]

        assert client.get("/api/v1/cameras/worker-config").status_code == 401
        assert (
            client.get(
                "/api/v1/cameras/worker-config",
                headers={"X-Edge-Relay-Token": "wrong"},
            ).status_code
            == 403
        )
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"Authorization": "Bearer relay-token"},
        )
        assert worker_config.status_code == 200
        assert worker_config.json() == {
            "registry_version": 1,
            "cameras": [
                {
                    "camera_id": camera["id"],
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://user:secret@camera.local:8554/live",
                }
            ],
        }

        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        assert relay_config.status_code == 200
        assert relay_config.json() == worker_config.json()

        patched = client.patch(
            f"/api/v1/cameras/{camera['id']}",
            headers=AUTH,
            json={"label": "Lobby North"},
        )
        assert patched.status_code == 200
        assert client.get("/api/v1/cameras", headers=AUTH).json()["registry_version"] == 2

        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)
        assert tested.status_code == 200
        assert tested.json() == {"ok": False, "error_class": "timeout"}

        deleted = client.delete(f"/api/v1/cameras/{camera['id']}", headers=AUTH)
        assert deleted.status_code == 204
        after_delete = client.get("/api/v1/cameras", headers=AUTH).json()
        assert after_delete == {"registry_version": 3, "cameras": []}

    captured_body = captured[0]["body"]
    assert isinstance(captured_body, dict)
    assert captured[0] == {
        "url": "http://backend/api/v1/edge/cameras",
        "method": "PUT",
        "authorization": "Bearer facility-token",
        "facility_id": "facility-1",
        "body": {
            "edge_camera_ref": captured_body["edge_camera_ref"],
            "label": "Lobby",
            "spaceId": "space-1",
        },
        "timeout": 0.5,
    }
    assert probe_calls == [
        {
            "url": "http://worker.local:8090/probe",
            "method": "POST",
            "relay_token": "relay-token",
            "body": {"rtsp_url": "rtsp://user:secret@camera.local:8554/live"},
            "timeout": 5.0,
        },
        {
            "url": "http://worker.local:8090/probe",
            "method": "POST",
            "relay_token": "relay-token",
            "body": {"rtsp_url": "rtsp://user:secret@camera.local:8554/live"},
            "timeout": 5.0,
        },
    ]


def test_worker_config_uses_registry_first_and_metadata_from_backend_pull(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.config_version = 42
    app.state.restart_epoch = 5
    app.state.pulled_config = PulledWorkerConfig(
        config_version=42,
        restart_epoch=5,
        night_window=PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        cameras=(
            PulledCameraConfig(
                camera_id="pulled-camera",
                space_id="space-pulled",
                label="Pulled",
                rtsp_url="rtsp://pulled/stream",
                online=True,
            ),
        ),
    )
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    expected = {
        "registry_version": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "local-facility",
                "rtsp_url": "rtsp://camera/stream",
            }
        ],
        "config_version": 42,
        "restart_epoch": 5,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
    }
    assert worker_config.json() == expected
    assert relay_config.status_code == 200
    assert relay_config.json() == expected


def test_worker_config_emits_default_camera_fps_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_CAMERA_FPS", "15")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["fps"] == 15.0
    assert camera["camera_id"] == "camera-1"


def test_worker_config_normalizes_pulled_cameras_when_registry_empty(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.config_version = 42
    app.state.restart_epoch = 5
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        cameras=(
            PulledCameraConfig(
                camera_id="pulled-camera",
                space_id="space-pulled",
                label="Pulled",
                rtsp_url="rtsp://pulled/stream",
                online=True,
            ),
            PulledCameraConfig(
                camera_id="pulled-without-rtsp",
                space_id="space-empty",
                label="Empty",
                rtsp_url=None,
                online=False,
            ),
        ),
    )
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    expected = {
        "registry_version": 0,
        "cameras": [
            {
                "camera_id": "pulled-camera",
                "facility_id": "local-facility",
                "rtsp_url": "rtsp://pulled/stream",
            }
        ],
        "config_version": 42,
        "restart_epoch": 5,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
    }
    assert worker_config.status_code == 200
    assert relay_config.status_code == 200
    assert worker_config.json() == expected
    assert relay_config.json() == expected
    assert set(worker_config.json()["cameras"][0]) == {"camera_id", "facility_id", "rtsp_url"}


def test_example_camera_registry_seed_is_loadable_and_sanitized() -> None:
    seed_path = REPO_ROOT / "backend" / "app" / "cameras.example.json"
    snapshot = CameraRegistryStore(seed_path).snapshot()

    assert snapshot["registry_version"] == 1
    cameras = snapshot["cameras"]
    assert isinstance(cameras, list)
    assert len(cameras) == 1
    loaded_record = cameras[0]
    assert isinstance(loaded_record, dict)
    record = {str(key): value for key, value in loaded_record.items()}
    assert record["id"] == "example-room-camera"
    assert record["rtsp_url"] == "rtsp://camera.example.invalid/trackID=1"

    public = public_camera(record)
    assert public == {
        "id": "example-room-camera",
        "label": "Example Room Camera",
        "rtsp_url_masked": "rtsp://redacted-camera/trackID=1",
        "space_id": "example-room",
        "backend_camera_id": None,
        "status": "unknown",
        "decode_backend": None,
        "created_at": "2026-01-01T00:00:00.000Z",
    }

    serialized = json.dumps(snapshot, sort_keys=True).lower()
    for forbidden in ("10.10.", "@", "admin", "password", "token"):
        assert forbidden not in serialized


def test_system_reports_backend_state_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BACKEND_URL", "http://backend")
    monkeypatch.setenv("ML_EDGE_VERSION", "2026.07.06")

    app = create_app(lifespan=no_lifespan)
    app.state.backend_reachable = True
    app.state.backend_last_ok_at = "2026-07-06T00:00:00.000Z"

    with TestClient(app) as client:
        response = client.get("/api/v1/system")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == {
        "configured": True,
        "reachable": True,
        "last_ok_at": "2026-07-06T00:00:00.000Z",
    }
    assert body["version"] == "2026.07.06"
    assert body["image_digests"] == {"ml_api": None, "ml_worker": None}
    assert set(body["storage"]["clip_store"]) == {"total_bytes", "used_bytes", "used_pct"}
    assert body["updated_at"].endswith("Z")


# test_config_pull_persists_lkg_and_falls_back (edge): LKG persist/fallback round-trip is
# superseded by tests/test_worker_config_pull_lkg.py (whole-file, worker-side pull/LKG/YAML
# coverage) and tests/test_worker_config_lifecycle.py
# ::test_fresh_pull_uses_auth_and_replaces_lkg_atomically /
# ::test_unreachable_backend_uses_stale_lkg_without_zeroing_cameras. The one assertion
# neither file makes is the fps/enabled_domains pull-mapping, ported below.
# test_restart_check_detects_registry_version_change (edge): superseded by
# tests/test_worker_restart_directive.py::test_restart_check_polls_immediately_on_first_call,
# ::test_restart_check_polls_again_once_interval_elapses, and
# ::test_tracker_observe_advances_current_on_strictly_greater_candidate.
def test_worker_config_pull_maps_fps_and_enabled_domains_from_relay_payload(
    tmp_path: Path,
) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    payload: JsonObject = {
        "registry_version": 9,
        "config_version": 9,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://camera/stream",
                "fps": 4,
                "domains": ["fall"],
            }
        ],
    }

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-token",
        store=store,
        urlopen=lambda _request, _timeout: FakeHTTPResponse(payload),
    )

    assert snapshot is not None
    assert snapshot.config.cameras[0].fps == 4
    assert snapshot.config.domains.enabled_domains == ("fall",)



def test_patch_pending_camera_preserves_local_id_after_backend_mapping(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera/stream",
                "space_id": "space-1",
            },
        )
        assert created.status_code == 201
        local_id = created.json()["id"]
        assert store.get(local_id)["mapping_pending"] is True

        # Existing clip manifests refer to this local id and must remain addressable.
        monkeypatch.setattr(
            "backend.app.features.cameras.router._map_backend",
            lambda *_args, **_kwargs: MappingResult(
                backend_camera_id="backend-camera-1", pending=False, reachable=True
            ),
        )
        response = client.patch(
            f"/api/v1/cameras/{local_id}",
            headers=AUTH,
            json={"label": "Lobby North"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == local_id
    assert response.json()["backend_camera_id"] == "backend-camera-1"
    assert response.json()["mapping_pending"] is False
    assert store.get(local_id)["backend_camera_id"] == "backend-camera-1"
    assert store.get("backend-camera-1") is None

def test_patch_camera_sets_decode_backend_and_worker_config_emits_it(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"decode_backend": "NVDEC"},
        )
        assert patched.status_code == 200
        assert patched.json()["decode_backend"] == "nvdec"

        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "nvdec"
    assert camera["camera_id"] == "camera-1"


def test_patch_camera_rejects_invalid_decode_backend(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"decode_backend": "gstreamer"},
        )

    assert patched.status_code == 400


def test_create_camera_rejects_invalid_decode_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera.local/live",
                "decode_backend": "gstreamer",
            },
        )

    assert created.status_code == 400


def test_worker_config_emits_default_decode_backend_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_DECODE_BACKEND", "cpu")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "cpu"
    assert camera["camera_id"] == "camera-1"


def test_worker_config_prefers_record_decode_backend_over_env_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_DECODE_BACKEND", "cpu")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        decode_backend="nvdec",
    )

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "nvdec"
def test_list_cameras_includes_backend_only_roster_camera(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-101",
                label="Room 101",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["cameras"] == [
        {
            "id": "backend-1",
            "label": "Room 101",
            "rtsp_url_masked": "rtsp://***",
            "space_id": "space-101",
            "backend_camera_id": "backend-1",
            "mapping_pending": True,
            "status": "unknown",
            "decode_backend": None,
            "created_at": "2026-07-10T00:00:00.000Z",
            "space_name": "101호",
            "floor_name": "1층",
        }
    ]
def test_list_cameras_includes_backend_only_roster_camera_without_created_at(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-no-created-at",
                space_id="space-101",
                label="Room 101",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at=None,
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["cameras"] == [
        {
            "id": "backend-no-created-at",
            "label": "Room 101",
            "rtsp_url_masked": "rtsp://***",
            "space_id": "space-101",
            "backend_camera_id": "backend-no-created-at",
            "mapping_pending": True,
            "status": "unknown",
            "decode_backend": None,
            "created_at": None,
            "space_name": "101호",
            "floor_name": "1층",
        }
    ]




def test_list_cameras_includes_local_only_camera_with_null_roster_names(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="local-1",
        label="Local camera",
        rtsp_url="rtsp://local/stream",
        space_id="local-space",
        status="online",
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["id"] == "local-1"
    assert camera["space_name"] is None
    assert camera["floor_name"] is None
    assert camera["mapping_pending"] is False


def test_list_cameras_joins_local_transport_with_backend_roster_metadata(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="local-1",
        label="Local label",
        rtsp_url="rtsp://user:secret@local/stream",
        space_id="local-space",
        status="online",
        backend_camera_id="backend-1",
        decode_backend="nvdec",
    )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="backend-space",
                label="Backend label",
                rtsp_url="rtsp://backend/stream",
                online=False,
                space_name="Backend room",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["id"] == "local-1"
    assert camera["label"] == "Backend label"
    assert camera["space_id"] == "backend-space"
    assert camera["created_at"] == "2026-07-10T00:00:00.000Z"
    assert camera["space_name"] == "Backend room"
    assert camera["floor_name"] == "2F"
    assert camera["rtsp_url_masked"] == "rtsp://***:***@redacted-camera/stream"
    assert camera["decode_backend"] == "nvdec"
    assert camera["status"] == "online"


def test_list_cameras_joins_unmapped_local_camera_by_unambiguous_space(tmp_path) -> None:
    """A local camera without backend_camera_id joins the sole same-space roster row.

    Local registrations predating the backend mapping carry a space_id only;
    cameras never move between rooms (spec R17), so a single roster camera in
    the same space is the same physical camera and must not appear twice.
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="local-1",
        label="Local label",
        rtsp_url="rtsp://user:secret@local/stream",
        space_id="space-1",
        status="online",
        backend_camera_id=None,
        decode_backend="nvdec",
    )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-1",
                label="Backend label",
                rtsp_url=None,
                online=True,
                space_name="205호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
            PulledCameraConfig(
                camera_id="backend-2",
                space_id="space-2",
                label="Other room",
                rtsp_url=None,
                online=True,
                space_name="202호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    joined = [c for c in cameras if c["id"] == "local-1"]
    assert len(joined) == 1
    assert joined[0]["space_name"] == "205호"
    assert joined[0]["backend_camera_id"] == "backend-1"
    # The joined roster row must not also appear as a backend-only duplicate.
    assert not any(c["id"] == "backend-1" for c in cameras)
    # The unrelated roster camera still appears backend-only.
    assert any(c["id"] == "backend-2" and c["status"] == "unknown" for c in cameras)


@pytest.mark.parametrize("unmapped_first", [True, False])
def test_space_fallback_never_steals_explicitly_mapped_roster_row(
    tmp_path, unmapped_first: bool
) -> None:
    """An unmapped local sibling must not consume a roster row owned by an
    explicitly mapped local camera, regardless of registry iteration order."""
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    order = (
        ("local-unmapped", "local-mapped") if unmapped_first else ("local-mapped", "local-unmapped")
    )
    for camera_id in order:
        store.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url="rtsp://user:secret@local/stream",
            space_id="space-1",
            status="online",
            backend_camera_id="backend-1" if camera_id == "local-mapped" else None,
            decode_backend="nvdec",
        )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-1",
                label="Backend label",
                rtsp_url=None,
                online=True,
                space_name="205호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = {c["id"]: c for c in response.json()["cameras"]}
    # The explicit mapping owns the roster metadata...
    assert cameras["local-mapped"]["space_name"] == "205호"
    assert cameras["local-mapped"]["backend_camera_id"] == "backend-1"
    # ...and the unmapped sibling never steals it.
    assert cameras["local-unmapped"]["space_name"] is None
    assert cameras["local-unmapped"]["backend_camera_id"] is None
    backend_ids = [c.get("backend_camera_id") for c in cameras.values()]
    assert backend_ids.count("backend-1") == 1

@pytest.mark.parametrize(
    ("local_ids", "backend_ids"),
    (
        (("local-1", "local-2"), ("backend-1",)),
        (("local-2", "local-1"), ("backend-1",)),
        (("local-1",), ("backend-1", "backend-2")),
        (("local-1",), ("backend-2", "backend-1")),
    ),
)
def test_list_cameras_does_not_join_ambiguous_space(
    tmp_path, local_ids: tuple[str, ...], backend_ids: tuple[str, ...]
) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    for camera_id in local_ids:
        store.create(
            camera_id=camera_id,
            label=f"Local {camera_id}",
            rtsp_url=f"rtsp://{camera_id}/stream",
            space_id="space-1",
            status="online",
        )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=tuple(
            PulledCameraConfig(
                camera_id=camera_id,
                space_id="space-1",
                label=f"Backend {camera_id}",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at="2026-07-10T00:00:00.000Z",
            )
            for camera_id in backend_ids
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    local_cameras = [camera for camera in cameras if camera["id"] in local_ids]
    assert len(local_cameras) == len(local_ids)
    assert all(camera["backend_camera_id"] is None for camera in local_cameras)
    assert all(camera["space_name"] is None for camera in local_cameras)
    assert {camera["id"] for camera in cameras} == {*local_ids, *backend_ids}


def test_list_cameras_reflects_room_name_after_roster_refresh(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [
        {
            "configVersion": 1,
            "cameras": [
                {
                    "id": "backend-1",
                    "spaceId": "space-1",
                    "label": "Room 101",
                    "rtspUrl": None,
                    "online": True,
                    "spaceName": "101호",
                    "floorName": "1층",
                    "createdAt": "2026-07-10T00:00:00.000Z",
                }
            ],
        },
        {
            "configVersion": 2,
            "cameras": [
                {
                    "id": "backend-1",
                    "spaceId": "space-1",
                    "label": "Room 101",
                    "rtspUrl": None,
                    "online": True,
                    "spaceName": "새 101호",
                    "floorName": "1층",
                    "createdAt": "2026-07-10T01:00:00.000Z",
                }
            ],
        },
    ]
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        payload = configs[calls["count"]]
        calls["count"] += 1
        return FakeHTTPResponse(payload)

    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend/ml-config")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")

    assert refresh_backend_config(app) is True
    with TestClient(app) as client:
        assert client.get("/api/v1/cameras", headers=AUTH).json()["cameras"][0][
            "space_name"
        ] == "101호"

        assert refresh_backend_config(app) is True
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.json()["cameras"][0]["space_name"] == "새 101호"


def test_roster_refresh_failure_preserves_last_good_and_marks_stale(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeHTTPResponse(
                {
                    "configVersion": 9,
                    "cameras": [
                        {
                            "id": "backend-1",
                            "spaceId": "space-1",
                            "label": "Room 101",
                            "rtspUrl": None,
                            "online": True,
                            "createdAt": "2026-07-10T00:00:00.000Z",
                        }
                    ],
                }
            )
        raise urllib.error.URLError("offline")

    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend/ml-config")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)

    assert refresh_backend_config(app) is True
    received_at = app.state.backend_roster["received_at"]
    assert refresh_backend_config(app) is False
    assert app.state.pulled_config.config_version == 9
    assert app.state.backend_roster == {
        "config_version": 9,
        "received_at": received_at,
        "stale": True,
    }
