from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.main import create_app, no_lifespan
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

CAP = 65_536
OVERSIZED = (413, b'{"detail":"request body exceeds maximum of 65536 bytes"}')
JSON_HEADERS = {"Content-Type": "application/json"}


def _dashboard_body_routes() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for route in create_app(lifespan=no_lifespan).routes:
        if not isinstance(route, APIRoute) or route.body_field is None:
            continue
        if route.path.startswith("/api/v1/relay/"):
            continue
        path = re.sub(r"\{[^}]+\}", "x1", route.path)
        found.extend((method, path) for method in sorted(route.methods))
    return found


DASHBOARD_BODY_ROUTES = _dashboard_body_routes()


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> FastAPI:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.dashboard_username = "operator"
    app.state.dashboard_password = "correct horse"
    return app


def _chunks(total: int, *, chunk: int = 16 * 1024) -> Iterator[bytes]:
    sent = 0
    while sent < total:
        step = min(chunk, total - sent)
        sent += step
        yield b" " * step


def _padded(body: object, size: int) -> bytes:
    encoded = json.dumps(body).encode()
    return encoded + b" " * (size - len(encoded))


def test_login_and_dashboard_routes_are_all_covered() -> None:
    assert ("POST", "/api/v1/auth/session") in DASHBOARD_BODY_ROUTES
    assert ("PUT", "/api/v1/runtime-settings") in DASHBOARD_BODY_ROUTES
    assert len(DASHBOARD_BODY_ROUTES) >= 22


@pytest.mark.parametrize(("method", "path"), DASHBOARD_BODY_ROUTES)
def test_an_oversized_content_length_is_rejected_before_the_body_is_read(
    app: FastAPI, method: str, path: str
) -> None:
    with TestClient(app) as client:
        response = client.request(
            method,
            path,
            content=b"{}",
            headers={**JSON_HEADERS, "Content-Length": str(CAP + 1)},
        )
    assert (response.status_code, response.content) == OVERSIZED


@pytest.mark.parametrize(("method", "path"), DASHBOARD_BODY_ROUTES)
def test_an_oversized_chunked_body_is_rejected(app: FastAPI, method: str, path: str) -> None:
    with TestClient(app) as client:
        response = client.request(method, path, content=_chunks(CAP + 1), headers=JSON_HEADERS)
    assert (response.status_code, response.content) == OVERSIZED


def test_a_login_body_at_the_cap_still_logs_in(app: FastAPI) -> None:
    body = _padded({"username": "operator", "password": "correct horse"}, CAP)
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/session", content=body, headers=JSON_HEADERS)
    assert response.status_code == 204


def test_a_chunked_login_body_at_the_cap_still_logs_in(app: FastAPI) -> None:
    body = _padded({"username": "operator", "password": "correct horse"}, CAP)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/session",
            content=(body[i : i + 1000] for i in range(0, CAP, 1000)),
            headers=JSON_HEADERS,
        )
    assert response.status_code == 204


def test_an_invalid_body_at_the_cap_is_still_validated_before_auth(app: FastAPI) -> None:
    body = _padded({"clip_export_enabled": "nope", "expected_version": 0}, CAP)
    with TestClient(app) as client:
        response = client.put("/api/v1/runtime-settings", content=body, headers=JSON_HEADERS)
    assert response.status_code == 422


def test_a_valid_body_at_the_cap_still_needs_a_session(app: FastAPI) -> None:
    body = _padded({"clip_export_enabled": True, "expected_version": 0}, CAP)
    with TestClient(app) as client:
        response = client.put("/api/v1/runtime-settings", content=body, headers=JSON_HEADERS)
    assert response.status_code == 401
