from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan
from backend.app.shared.dashboard_sessions import (
    DashboardSessionStore,
    PlaintextDashboardCredentials,
)

RELAY_HEADERS = {"X-Edge-Relay-Token": "worker-secret", "Content-Type": "application/json"}


@pytest.fixture
def app() -> FastAPI:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "worker-secret"
    app.state.dashboard_sessions = DashboardSessionStore(
        credentials=PlaintextDashboardCredentials(username="operator", password="pw")
    )
    return app


def _chunks(total: int) -> Iterator[bytes]:
    sent = 0
    while sent < total:
        step = min(4096, total - sent)
        sent += step
        yield b"a" * step


@pytest.mark.parametrize(
    ("path", "cap"),
    [
        ("/api/v1/relay/alerts", 524_288),
        ("/api/v1/relay/heartbeat", 4_096),
        ("/api/v1/relay/runtime-status", 65_536),
        ("/api/v1/relay/snapshot-attachments", 8_192),
        ("/api/v1/relay/snapshot-dispositions", 8_192),
    ],
)
def test_relay_streams_stop_at_their_own_cap(app: FastAPI, path: str, cap: int) -> None:
    with TestClient(app) as client:
        over = client.post(path, headers=RELAY_HEADERS, content=_chunks(cap + 1))
        at_cap = client.post(path, headers=RELAY_HEADERS, content=_chunks(cap))

    assert over.status_code == 413
    assert over.json() == {"detail": f"request body exceeds maximum of {cap} bytes"}
    assert at_cap.status_code == 422
    assert at_cap.json()["detail"][0]["type"] == "json_invalid"


def test_evidence_clip_route_has_no_relay_body_cap(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.put(
            "/api/v1/relay/clips/clip-1", headers=RELAY_HEADERS, content=_chunks(2 * 1_048_576)
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


@pytest.mark.parametrize("path", ["/api/v1/connection", "/api/v1/connection/topology-preview"])
def test_connection_routes_require_a_dashboard_session(app: FastAPI, path: str) -> None:
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "dashboard session required"}
