from __future__ import annotations

from fastapi.testclient import TestClient
from test_api_ingest_relay import FakeBackendIngestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.cameras.router import _mapping_state
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_LOCAL_ID = "11111111-2222-3333-4444-555555555555"
_HUB_ID = "cmsnvr-abc123"
_RELAY_HEADERS = {"X-Edge-Relay-Token": "relay-token"}


def _app_with_camera(
    sandbox: ProductSandbox,
    audit_runtime: PostgresAuditRuntime,
    *,
    backend_camera_id: str | None,
    pending: bool = False,
):
    app = postgres_api_app(sandbox, audit_runtime)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry
    store.create(
        camera_id=_LOCAL_ID,
        label="Room 101",
        rtsp_url="rtsp://example/room-101",
        space_id=None,
        status="online",
        backend_camera_id=backend_camera_id,
        mapping_pending=pending,
    )
    return app, store


def _alert_body(camera_id: str) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "facility_id": "facility-1",
        "event_type": "bed-exit",
        "detected_at": "2026-08-17T10:00:00.000Z",
        "probability": 0.97,
    }


def test_relay_alert_never_egresses_local_id_for_unmapped_camera(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    app, _ = _app_with_camera(
        postgres_product_sandbox, postgres_audit_runtime, backend_camera_id=None
    )
    fake = FakeBackendIngestClient()
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts", json=_alert_body(_LOCAL_ID), headers=_RELAY_HEADERS
        )

    assert response.status_code == 202
    assert _LOCAL_ID not in fake.egress_camera_ids
    assert fake.alerts == []


def test_relay_alert_egresses_hub_id_when_mapped(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    app, _ = _app_with_camera(
        postgres_product_sandbox, postgres_audit_runtime, backend_camera_id=_HUB_ID
    )
    fake = FakeBackendIngestClient()
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts", json=_alert_body(_LOCAL_ID), headers=_RELAY_HEADERS
        )

    assert response.status_code == 202
    assert fake.egress_camera_ids == [_HUB_ID]
    assert _LOCAL_ID not in fake.egress_camera_ids
    assert len(fake.alerts) == 1


def test_worker_config_still_serves_unmapped_camera(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    app, _ = _app_with_camera(
        postgres_product_sandbox, postgres_audit_runtime, backend_camera_id=None
    )
    app.state.backend_ingest_client = FakeBackendIngestClient()

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=_RELAY_HEADERS)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    assert len(cameras) == 1, "an unmapped camera must still reach the worker"
    assert cameras[0]["rtsp_url"] == "rtsp://example/room-101"


def test_mapping_state_distinguishes_pending_from_unmapped() -> None:
    assert _mapping_state({"backend_camera_id": _HUB_ID}) == "mapped"
    assert _mapping_state({"backend_camera_id": None, "mapping_pending": True}) == "pending"
    assert _mapping_state({"backend_camera_id": None, "mapping_pending": False}) == "unmapped"
    assert _mapping_state({}) == "unmapped"
    assert _mapping_state({"backend_camera_id": "   "}) == "unmapped"
