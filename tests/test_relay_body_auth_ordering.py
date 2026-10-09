from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.relay import router as relay_router
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> FastAPI:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = "worker-secret"
    return app


@pytest.fixture
def app_with_camera(app: FastAPI) -> FastAPI:
    app.state.camera_registry.create(
        camera_id="cam-1",
        label="cam",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    return app


def test_oversized_content_length_is_rejected_before_body_parse(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
                "Content-Length": str(relay_router.MAX_RELAY_REQUEST_BODY_BYTES + 1),
            },
            content=b"{}",
        )
    assert response.status_code == 413


def test_missing_relay_token_is_rejected_without_accepting_payload(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json={
                "event_type": "fall",
                "probability": 0.9,
                "detected_at": "2026-08-14T00:00:00Z",
                "camera_id": "cam-1",
                "facility_id": "fac-1",
            },
        )
    assert response.status_code == 401


def test_authorized_small_heartbeat_is_accepted(app_with_camera: FastAPI) -> None:
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"X-Edge-Relay-Token": "worker-secret"},
            json={"camera_id": "cam-1", "facility_id": "fac-1"},
        )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_authorize_relay_non_ascii_token_compares_constant_time_without_typeerror() -> None:
    from backend.app.shared.http.relay_http import authorize_relay

    state = SimpleNamespace(edge_relay_token="중계-토큰")
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    authorize_relay(request, "중계-토큰")

    with pytest.raises(HTTPException) as exc_info:
        authorize_relay(request, "wrong-token")
    assert exc_info.value.status_code == 403


def _oversized_chunks(total_bytes: int, *, chunk: int = 512) -> Iterator[bytes]:
    sent = 0
    while sent < total_bytes:
        step = min(chunk, total_bytes - sent)
        sent += step
        yield b"a" * step


def test_chunked_oversized_body_without_content_length_is_rejected(
    app_with_camera: FastAPI,
) -> None:
    over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
            },
            content=_oversized_chunks(over),
        )
    assert response.status_code == 413


def test_chunked_body_without_content_length_is_accepted(app_with_camera: FastAPI) -> None:
    body = json.dumps({"camera_id": "cam-1", "facility_id": "fac-1"}).encode("utf-8")

    def _stream() -> Iterator[bytes]:
        yield body[: len(body) // 2]
        yield body[len(body) // 2 :]

    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
            },
            content=_stream(),
        )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_unauthorized_within_limit_body_is_rejected_before_pydantic_parse(
    app_with_camera: FastAPI,
) -> None:
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"Content-Type": "application/json"},
            json={"camera_id": "cam-1", "facility_id": "fac-1"},
        )
    assert response.status_code == 401


def test_unauthorized_oversized_chunked_body_is_rejected_at_transport_bound(
    app_with_camera: FastAPI,
) -> None:
    over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"Content-Type": "application/json"},
            content=_oversized_chunks(over),
        )
    assert response.status_code == 413


class _StartupSignalServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.listening = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            self.listening.set()


class _LiveApp:
    def __init__(self, app: FastAPI) -> None:
        self.port = _free_tcp_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = _StartupSignalServer(config)
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="relay-bounded-read"
        )
        self._thread.start()
        if not self._server.listening.wait(timeout=10.0):
            pytest.fail("timed out waiting for relay uvicorn startup")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


def test_real_uvicorn_no_content_length_oversized_is_rejected(app_with_camera: FastAPI) -> None:
    server = _LiveApp(app_with_camera)
    try:
        over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
        with httpx.Client(base_url=server.base_url) as client:
            request = client.build_request(
                "POST",
                "/api/v1/relay/heartbeat",
                headers={
                    "X-Edge-Relay-Token": "worker-secret",
                    "Content-Type": "application/json",
                },
                content=_oversized_chunks(over),
            )
            assert "content-length" not in request.headers
            assert request.headers.get("transfer-encoding") == "chunked"
            response = client.send(request)
    finally:
        server.stop()
    assert response.status_code == 413


def test_real_uvicorn_no_content_length_within_limit_is_accepted(app_with_camera: FastAPI) -> None:
    server = _LiveApp(app_with_camera)
    body = json.dumps({"camera_id": "cam-1", "facility_id": "fac-1"}).encode("utf-8")

    def _stream() -> Iterator[bytes]:
        yield body[:3]
        yield body[3:]

    try:
        with httpx.Client(base_url=server.base_url) as client:
            request = client.build_request(
                "POST",
                "/api/v1/relay/heartbeat",
                headers={
                    "X-Edge-Relay-Token": "worker-secret",
                    "Content-Type": "application/json",
                },
                content=_stream(),
            )
            assert "content-length" not in request.headers
            response = client.send(request)
    finally:
        server.stop()
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
