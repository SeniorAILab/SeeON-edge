from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan


def _client(tmp_path):
    app = create_app(lifespan=no_lifespan)
    app.state.dashboard_username = "operator"
    app.state.dashboard_password = "correct horse"
    app.state.edge_relay_token = "worker-secret"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")
    return TestClient(app)


def test_dashboard_login_uses_httponly_cookie_for_protected_routes(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )

        assert response.status_code == 204
        cookie = response.headers["set-cookie"]
        assert "ml_dashboard_session=" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert client.get("/api/v1/cameras").status_code == 200


def test_dashboard_rejects_invalid_login_and_worker_token(tmp_path) -> None:
    with _client(tmp_path) as client:
        invalid = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "wrong"},
        )
        worker_token = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

        assert invalid.status_code == 401
        assert worker_token.status_code == 401


def test_worker_config_keeps_dedicated_relay_auth_when_dashboard_sessions_are_enabled(
    tmp_path,
) -> None:
    with _client(tmp_path) as client:
        accepted = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "worker-secret"},
        )
        dashboard_route = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

        assert accepted.status_code == 200
        assert dashboard_route.status_code == 401


def test_dashboard_routes_fail_closed_without_credentials_or_explicit_legacy_opt_in(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("API_ALLOW_LEGACY_DASHBOARD_AUTH")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "worker-secret"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

    assert response.status_code == 503


def test_concurrent_first_logins_share_one_session_store(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.dashboard_username = "operator"
    app.state.dashboard_password = "correct horse"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "cameras.json")

    def login_and_query(_index: int) -> tuple[int, int]:
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "correct horse"},
            )
            query = client.get("/api/v1/cameras")
            return login.status_code, query.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(login_and_query, range(16)))

    assert statuses == [(204, 200)] * 16


def test_dashboard_logout_revokes_session(tmp_path) -> None:
    with _client(tmp_path) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )
        assert login.status_code == 204
        assert client.delete("/api/v1/auth/session").status_code == 204

        assert client.get("/api/v1/cameras").status_code == 401
