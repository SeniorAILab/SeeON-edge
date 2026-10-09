from __future__ import annotations

import json

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_HEADERS, relay_postgres_app

pytest_plugins = ("tests_support.postgres_sandbox",)

ALERT = {
    "event_type": "fall",
    "probability": 0.5,
    "detected_at": "2026-10-09T00:00:00Z",
    "camera_id": "camera-1",
    "facility_id": "facility-1",
}


class _Opaque:
    __slots__ = ()


class _ExplodingBody(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def explode(cls, value: str) -> str:
        if value == "runtime":
            raise RuntimeError("validator bug")
        raise PydanticCustomError("opaque", "opaque failure", {"thing": _Opaque()})


@pytest.fixture
def client(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> TestClient:
    app = relay_postgres_app(postgres_product_sandbox, postgres_audit_runtime)
    router = APIRouter()

    @router.post("/__explode")
    def explode(body: _ExplodingBody) -> None:
        del body

    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _post(client: TestClient, path: str, body: object) -> tuple[int, bytes]:
    response = client.post(
        path,
        content=json.dumps(body, ensure_ascii=True, allow_nan=True).encode("ascii"),
        headers={**RELAY_HEADERS, "Content-Type": "application/json"},
    )
    return response.status_code, response.content


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        (
            "/api/v1/relay/alerts",
            {**ALERT, "probability": float("nan")},
            (
                b'{"detail":[{"type":"less_than_equal","loc":["body","probability"],'
                b'"msg":"Input should be less than or equal to 1","input":"NaN","ctx":{"le":1.0}}]}'
            ),
        ),
        (
            "/api/v1/relay/alerts",
            {**ALERT, "probability": float("-inf")},
            (
                b'{"detail":[{"type":"greater_than_equal","loc":["body","probability"],'
                b'"msg":"Input should be greater than or equal to 0","input":"-Infinity",'
                b'"ctx":{"ge":0.0}}]}'
            ),
        ),
        (
            "/api/v1/relay/alerts",
            {**ALERT, "camera_id": "\ud800"},
            (
                b'{"detail":[{"type":"string_unicode","loc":["body","camera_id"],'
                b'"msg":"Input should be a valid string, '
                b'unable to parse raw data as a unicode string",'
                b'"input":"\\\\ud800"}]}'
            ),
        ),
        (
            "/api/v1/relay/alerts",
            {**ALERT, "evidence": "\ud800"},
            (
                b'{"detail":[{"type":"dict_type","loc":["body","evidence"],'
                b'"msg":"Input should be a valid dictionary","input":"\\\\ud800"}]}'
            ),
        ),
        (
            "/api/v1/relay/heartbeat",
            {"camera_id": "camera-1", "facility_id": "facility-1", "\ud800": 1},
            (
                b'{"detail":[{"type":"string_unicode","loc":["body"],'
                b'"msg":"Input should be a valid string, '
                b'unable to parse raw data as a unicode string",'
                b'"input":"\\\\ud800"}]}'
            ),
        ),
    ],
    ids=[
        "nan-probability",
        "negative-infinity",
        "surrogate-camera",
        "surrogate-evidence",
        "surrogate-key",
    ],
)
def test_unencodable_input_is_a_422_with_the_value_escaped(
    client: TestClient, path: str, body: object, expected: bytes
) -> None:
    assert _post(client, path, body) == (422, expected)


def test_encodable_validation_errors_keep_their_bytes(client: TestClient) -> None:
    assert _post(client, "/api/v1/relay/alerts", {**ALERT, "evidence": "é", "probability": 2}) == (
        422,
        (
            b'{"detail":[{"type":"less_than_equal","loc":["body","probability"],'
            b'"msg":"Input should be less than or equal to 1","input":2,"ctx":{"le":1.0}},'
            b'{"type":"dict_type","loc":["body","evidence"],'
            b'"msg":"Input should be a valid dictionary","input":"\xc3\xa9"}]}'
        ),
    )


def test_a_non_validation_error_raised_while_validating_stays_a_500(client: TestClient) -> None:
    assert _post(client, "/__explode", {"value": "runtime"}) == (
        500,
        b'{"detail":"internal server error"}',
    )


def test_a_failure_while_rendering_that_is_not_about_encoding_stays_a_500(
    client: TestClient,
) -> None:
    assert _post(client, "/__explode", {"value": "opaque"}) == (
        500,
        b'{"detail":"internal server error"}',
    )
