from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def test_models_reports_gateway_metadata_only() -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.camera_inventory = {"camera-1": {"camera_id": "camera-1"}}
    app.state.backend_ingest_client = object()

    response = TestClient(app).get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ml-api",
        "role": "gateway",
        "ml": "external-worker",
        "relay": {"backend_configured": True, "camera_count": 1},
    }
