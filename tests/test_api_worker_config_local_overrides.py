from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

AUTH = {"Authorization": "Bearer relay-token"}
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_FACILITY_ID", raising=False)
    monkeypatch.delenv("ML_API_DETECTION_TZ", raising=False)


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def test_with_no_local_overrides_the_response_reflects_the_externally_pulled_state(
    app: FastAPI,
) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=3,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="Asia/Seoul"),
        cameras=(),
        detection_windows={
            "bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="Asia/Seoul"),
            "fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul"),
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["night_window"] == {"start": "22:00", "end": "06:00", "tz": "Asia/Seoul"}
    assert body["detection_windows"] == {
        "bed_exit": {"start": "22:00", "end": "06:00", "tz": "Asia/Seoul"},
        "fall": {"start": "08:00", "end": "20:00", "tz": "Asia/Seoul"},
    }
    assert "domains" not in body
    assert "clip_store_subdir" not in body


def test_local_window_setting_overrides_the_pulled_window_and_reuses_its_tz(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={"fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul")},
    )

    with TestClient(app) as client:
        _login(client)
        put_response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "window", "start": "09:00", "end": "18:00"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        assert put_response.status_code == 200
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["detection_windows"]["fall"] == {
        "start": "09:00",
        "end": "18:00",
        "tz": "Asia/Seoul",
    }
    assert body["domains"] == {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}
    assert "bed_exit" not in body["detection_windows"]
    assert "night_window" not in body


def test_local_always_on_setting_removes_any_pulled_window_for_that_domain(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        cameras=(),
        detection_windows={"bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC")},
    )

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    body = response.json()
    assert "detection_windows" not in body
    assert "night_window" not in body
    assert body["domains"] == {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}


def test_local_off_setting_disables_the_domain_and_drops_its_window_and_alias(
    app: FastAPI,
) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        cameras=(),
        detection_windows={"bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC")},
    )

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": False, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    body = response.json()
    assert body["domains"]["bed_exit"] == {"enabled": False}
    assert "detection_windows" not in body
    assert "night_window" not in body


def test_clip_store_subdir_is_absent_until_a_non_root_location_is_selected(app: FastAPI) -> None:
    with TestClient(app) as client:
        before = client.get("/api/v1/cameras/worker-config", headers=AUTH)
        assert "clip_store_subdir" not in before.json()

        _login(client)
        put_response = client.put("/api/v1/clips/storage/location", json={"path": ""})
        assert put_response.status_code in (200, 404)


def test_clip_store_subdir_appears_once_a_selection_is_persisted_directly(app: FastAPI) -> None:
    app.state.clip_storage_location_store.put("external-drive")

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["clip_store_subdir"] == "external-drive"


def test_no_local_overrides_leaves_config_version_unchanged_from_pulled(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=2,
        night_window=None,
        cameras=(),
        detection_windows={},
    )
    app.state.config_version = 7
    app.state.restart_epoch = 2

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["config_version"] == 7


def test_local_overrides_present_move_config_version_away_from_pulled(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=2,
        night_window=None,
        cameras=(),
        detection_windows={},
    )
    app.state.config_version = 7
    app.state.restart_epoch = 2

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    body = response.json()
    assert body["config_version"] != 7


def test_same_overrides_saved_twice_yield_an_identical_config_version(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=2,
        night_window=None,
        cameras=(),
        detection_windows={},
    )
    app.state.config_version = 7
    app.state.restart_epoch = 2
    payload = {
        "domains": {
            "fall": {"on": True, "mode": "window", "start": "09:00", "end": "18:00"},
            "bed_exit": {"on": True, "mode": "always"},
        }
    }

    with TestClient(app) as client:
        _login(client)
        client.put("/api/v1/detection-settings", json=payload)
        first = client.get("/api/v1/cameras/worker-config", headers=AUTH).json()
        second = client.get("/api/v1/cameras/worker-config", headers=AUTH).json()
        client.put("/api/v1/detection-settings", json=payload)
        third = client.get("/api/v1/cameras/worker-config", headers=AUTH).json()

    assert first["config_version"] == second["config_version"] == third["config_version"]


def test_different_override_content_yields_a_different_config_version(app: FastAPI) -> None:
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=2,
        night_window=None,
        cameras=(),
        detection_windows={},
    )
    app.state.config_version = 7
    app.state.restart_epoch = 2

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        first = client.get("/api/v1/cameras/worker-config", headers=AUTH).json()

        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": False, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        second = client.get("/api/v1/cameras/worker-config", headers=AUTH).json()

    assert first["config_version"] != second["config_version"]


def test_worker_config_route_requires_relay_authorization(app: FastAPI) -> None:
    with TestClient(app) as client:
        unauthenticated = client.get("/api/v1/cameras/worker-config")
        wrong_token = client.get(
            "/api/v1/cameras/worker-config",
            headers={"Authorization": "Bearer wrong"},
        )

    assert unauthenticated.status_code == 401
    assert wrong_token.status_code == 403
