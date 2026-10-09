from __future__ import annotations

import json
import logging

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from shared.events.execution_records import WireBatch, WireGap, WireProvenance, WireRecord
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_HEADERS, relay_postgres_app

pytest_plugins = ("tests_support.postgres_sandbox", "tests_support.postgres_diagnostics_sandbox")

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
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> TestClient:
    app = relay_postgres_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.backend_build_revision = "backend-rev-1"
    app.state.execution_record_store = ExecutionRecordStore(
        postgres_diagnostics_sandbox.database, RetentionBudget(total_bytes=2**20)
    )
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
            "/api/v1/relay/alerts",
            {**ALERT, "evidence": ["\ud800"]},
            (
                b'{"detail":[{"type":"dict_type","loc":["body","evidence"],'
                b'"msg":"Input should be a valid dictionary","input":["\\\\ud800"]}]}'
            ),
        ),
        (
            "/api/v1/relay/alerts",
            {**ALERT, "evidence": float("nan")},
            (
                b'{"detail":[{"type":"dict_type","loc":["body","evidence"],'
                b'"msg":"Input should be a valid dictionary","input":"NaN"}]}'
            ),
        ),
        (
            "/api/v1/relay/alerts",
            {**ALERT, "facility_id": "\ud800", "evidence": {"domain": 5}},
            (
                b'{"detail":[{"type":"string_unicode","loc":["body","facility_id"],'
                b'"msg":"Input should be a valid string, '
                b'unable to parse raw data as a unicode string",'
                b'"input":"\\\\ud800"}]}'
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
        "surrogate-evidence-list",
        "nan-evidence",
        "surrogate-facility",
        "surrogate-key",
    ],
)
def test_unencodable_input_is_a_422_with_the_value_escaped(
    client: TestClient, path: str, body: object, expected: bytes
) -> None:
    assert _post(client, path, body) == (422, expected)


_PROVENANCE = WireProvenance(*(f"provenance-{index}" for index in range(7)))
_RECORD = WireRecord(
    record_kind="sdk.frame",
    camera_id="camera-1",
    worker_boot_id="boot-1",
    source_generation=1,
    stream_epoch=1,
    producer="sdk",
    producer_sequence=0,
    observed_at_ns=1,
    time_quality="pts",
    causal_unit_id="unit-1",
    outcome="accepted",
    payload={},
    source_pts_ns=33,
)
_GAP = WireGap("sdk", 1, 2, 1, 2, 2, "lane-overflow", 1, 1)
_SURROGATE = "\ud800"


def _batch() -> dict[str, object]:
    batch = WireBatch("camera-1", "boot-1", _PROVENANCE, (_RECORD,), (_GAP,))
    loaded: dict[str, object] = json.loads(batch.encode())
    del loaded["batch_id"]
    for record in loaded["records"]:
        del record["record_id"]
    return loaded


def _unicode_errors(*locs: tuple[object, ...]) -> bytes:
    items = [
        b'{"type":"string_unicode","loc":'
        + json.dumps(["body", *loc], separators=(",", ":")).encode()
        + b',"msg":"Input should be a valid string, unable to parse raw data as a unicode string",'
        + b'"input":"\\\\ud800"}'
        for loc in locs
    ]
    return b'{"detail":[' + b",".join(items) + b"]}"


def _everywhere(field: str) -> object:
    def edit(body: dict[str, object]) -> None:
        body[field] = _SURROGATE
        body["records"][0][field] = _SURROGATE

    return edit


@pytest.mark.parametrize(
    ("edit", "locs"),
    [
        (_everywhere("camera_id"), [("camera_id",), ("records", 0, "camera_id")]),
        (_everywhere("worker_boot_id"), [("worker_boot_id",), ("records", 0, "worker_boot_id")]),
        (
            lambda body: body["records"][0].update(camera_id=_SURROGATE),
            [("records", 0, "camera_id")],
        ),
        (lambda body: body["records"][0].update(producer=_SURROGATE), [("records", 0, "producer")]),
        (
            lambda body: body["provenance"].update(model_digest=_SURROGATE),
            [("provenance", "model_digest")],
        ),
        (lambda body: body["gaps"][0].update(cause=_SURROGATE), [("gaps", 0, "cause")]),
    ],
    ids=[
        "batch-and-record-camera",
        "batch-and-record-boot",
        "record-camera",
        "record-producer",
        "provenance-model-digest",
        "gap-cause",
    ],
)
def test_execution_record_string_fields_with_a_surrogate_are_a_422(
    client: TestClient, edit: object, locs: list[tuple[object, ...]]
) -> None:
    body = _batch()
    edit(body)
    assert _post(client, "/api/v1/relay/execution-records", body) == (422, _unicode_errors(*locs))


def test_a_recursion_error_while_storing_a_deep_alert_stays_a_500(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    evidence: dict[str, object] = {}
    for _ in range(1000):
        evidence = {"a": evidence}
    with caplog.at_level(logging.ERROR, logger="backend.app.main"):
        status, body = _post(
            client,
            "/api/v1/relay/alerts",
            {
                **ALERT,
                "edge_event_id": "00000000-0000-4000-9000-000000000001",
                "evidence": evidence,
            },
        )
    assert (status, body) == (500, b'{"detail":"internal server error"}')
    assert "exception_class=RecursionError" in caplog.text


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
