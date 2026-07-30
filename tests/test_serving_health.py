from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def test_health_live_ok() -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_503_while_booting() -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["reason"] == "booting"


def test_health_ready_200_when_catalog_path_is_unwritable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CATALOG_STORE", "/var/lib/ml-api/unwritable-catalog.sqlite3")
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert not hasattr(app.state, "catalog_store")

def test_health_ready_200_after_gateway_lifespan_boot() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "status": "ready"}


def test_fastapi_lifespan_does_not_assemble_ml_runtime() -> None:
    app = create_app()

    with TestClient(app) as client:
        status_body = client.get("/api/v1/status").json()
        health_body = client.get("/api/v1/health").json()

    assert not hasattr(app.state, "runtime")
    assert not hasattr(app.state, "incident_manager")
    assert not hasattr(app.state, "device")
    assert not hasattr(app.state, "model_registry")
    assert not hasattr(app.state, "source_registry")
    assert not hasattr(app.state, "model")
    assert not hasattr(app.state, "fall_pipeline")
    assert status_body["cameras"] == {}
    assert "stale_after_sec" in status_body
    assert health_body["status"] == "ok"
    assert health_body["relay"]["camera_count"] == 0
