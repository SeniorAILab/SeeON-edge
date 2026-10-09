from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

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


_Edit = Callable[[dict[str, Any]], None]


def _batch() -> dict[str, Any]:
    batch = WireBatch("camera-1", "boot-1", _PROVENANCE, (_RECORD,), (_GAP,))
    loaded: dict[str, Any] = json.loads(batch.encode())
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


def _everywhere(field: str) -> _Edit:
    def edit(body: dict[str, Any]) -> None:
        body[field] = _SURROGATE
        body["records"][0][field] = _SURROGATE

    return edit


def _at(*path: str | int) -> _Edit:
    def edit(body: dict[str, Any]) -> None:
        target: Any = body
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = _SURROGATE

    return edit


def _extra_record_key(body: dict[str, Any]) -> None:
    body["records"][0][_SURROGATE] = 1


_RECORD_STRINGS = (
    "camera_id",
    "producer",
    "record_kind",
    "time_quality",
    "outcome",
    "reason",
    "causal_unit_id",
    "parent_record_id",
)
_SURROGATE_CASES = [
    pytest.param(
        _everywhere("camera_id"),
        [("camera_id",), ("records", 0, "camera_id")],
        id="batch-and-record-camera_id",
    ),
    pytest.param(
        _everywhere("worker_boot_id"),
        [("worker_boot_id",), ("records", 0, "worker_boot_id")],
        id="batch-and-record-worker_boot_id",
    ),
    *[
        pytest.param(_at("records", 0, name), [("records", 0, name)], id=f"record-{name}")
        for name in _RECORD_STRINGS
    ],
    *[
        pytest.param(_at("gaps", 0, name), [("gaps", 0, name)], id=f"gap-{name}")
        for name in ("cause", "producer")
    ],
    *[
        pytest.param(_at("provenance", name), [("provenance", name)], id=f"provenance-{name}")
        for name in WireProvenance.__slots__
    ],
    pytest.param(_extra_record_key, [("records", 0)], id="record-extra-key-name"),
]


@pytest.mark.parametrize(("edit", "locs"), _SURROGATE_CASES)
def test_execution_record_string_fields_with_a_surrogate_are_a_422(
    client: TestClient, edit: _Edit, locs: list[tuple[object, ...]]
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


def _nested(depth: int) -> object:
    value: object = "é"
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("depth", [973, 1200])
def test_a_recursion_error_while_rendering_a_deep_input_stays_a_500(
    client: TestClient, caplog: pytest.LogCaptureFixture, depth: int
) -> None:
    with caplog.at_level(logging.ERROR, logger="backend.app.main"):
        status, body = _post(client, "/api/v1/relay/alerts", {**ALERT, "camera_id": _nested(depth)})
    assert (status, body) == (500, b'{"detail":"internal server error"}')
    assert "exception_class=RecursionError" in caplog.text


def test_a_deep_input_below_the_recursion_limit_is_a_422(client: TestClient) -> None:
    assert _post(client, "/api/v1/relay/alerts", {**ALERT, "camera_id": _nested(900)}) == (
        422,
        b'{"detail":[{"type":"string_type","loc":["body","camera_id"],'
        b'"msg":"Input should be a valid string","input":'
        + b"[" * 900
        + b'"\xc3\xa9"'
        + b"]" * 900
        + b"}]}",
    )


@pytest.mark.parametrize(
    ("body", "escaped"),
    [
        (b"\xff\xfe", b"\\\\xff\\\\xfe"),
        (b'{"camera_id":"\xff"}', b'{\\"camera_id\\":\\"\\\\xff\\"}'),
    ],
    ids=["bare-bytes", "json-shaped-bytes"],
)
def test_a_non_json_body_with_invalid_utf8_is_a_422_with_the_bytes_escaped(
    client: TestClient, body: bytes, escaped: bytes
) -> None:
    response = client.post(
        "/api/v1/relay/alerts",
        content=body,
        headers={**RELAY_HEADERS, "Content-Type": "text/plain"},
    )
    assert (response.status_code, response.content) == (
        422,
        b'{"detail":[{"type":"model_attributes_type","loc":["body"],'
        b'"msg":"Input should be a valid dictionary or object to extract fields from",'
        b'"input":"' + escaped + b'"}]}',
    )


def test_a_surrogate_in_an_echoed_object_key_is_escaped(client: TestClient) -> None:
    assert _post(client, "/api/v1/relay/alerts", {**ALERT, "zz": {"\ud800": 1}}) == (
        422,
        (
            b'{"detail":[{"type":"extra_forbidden","loc":["body","zz"],'
            b'"msg":"Extra inputs are not permitted","input":{"\\\\ud800":1}}]}'
        ),
    )


def test_encodable_values_next_to_an_unencodable_one_keep_their_bytes(client: TestClient) -> None:
    body = {**ALERT, "camera_id": "\ud800", "evidence": "é"}
    assert _post(client, "/api/v1/relay/alerts", body) == (
        422,
        (
            b'{"detail":[{"type":"string_unicode","loc":["body","camera_id"],'
            b'"msg":"Input should be a valid string, '
            b'unable to parse raw data as a unicode string",'
            b'"input":"\\\\ud800"},'
            b'{"type":"dict_type","loc":["body","evidence"],'
            b'"msg":"Input should be a valid dictionary","input":"\xc3\xa9"}]}'
        ),
    )


_COLLIDING_KEYS = b'{"\\ud800":1,"\\\\ud800":2}'
_COLLIDING_INPUT = b'{"\\\\ud800":1}'


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        (
            "POST",
            "/api/v1/relay/execution-records",
            b'{"detail":['
            + b",".join(
                b'{"type":"missing","loc":["body","' + field + b'"],'
                b'"msg":"Field required","input":' + _COLLIDING_INPUT + b"}"
                for field in (b"camera_id", b"worker_boot_id", b"provenance")
            )
            + b"]}",
        ),
        (
            "PUT",
            "/api/v1/relay/clips/clip-1",
            b'{"detail":[{"type":"union_tag_not_found","loc":["body"],'
            b'"msg":"Unable to extract tag using discriminator \'state\'",'
            b'"input":' + _COLLIDING_INPUT + b',"ctx":{"discriminator":"\'state\'"}}]}',
        ),
    ],
    ids=["execution-records", "clips"],
)
def test_keys_that_collide_after_escaping_keep_the_first_value(
    client: TestClient, method: str, path: str, expected: bytes
) -> None:
    response = client.request(
        method,
        path,
        content=_COLLIDING_KEYS,
        headers={**RELAY_HEADERS, "Content-Type": "application/json"},
    )
    assert (response.status_code, response.content) == (422, expected)


def test_an_unencodable_input_too_large_to_escape_is_omitted(client: TestClient) -> None:
    body = {**ALERT, "zz": [*range(10_000), "\ud800"]}
    assert _post(client, "/api/v1/relay/alerts", body) == (
        422,
        (
            b'{"detail":[{"type":"extra_forbidden","loc":["body","zz"],'
            b'"msg":"Extra inputs are not permitted","input":"omitted: too large to escape"}]}'
        ),
    )


def test_an_unencodable_input_small_enough_to_escape_is_echoed(client: TestClient) -> None:
    body = {**ALERT, "zz": [*range(9_000), "\ud800"]}
    assert _post(client, "/api/v1/relay/alerts", body) == (
        422,
        (
            b'{"detail":[{"type":"extra_forbidden","loc":["body","zz"],'
            b'"msg":"Extra inputs are not permitted","input":['
            + ",".join(str(index) for index in range(9_000)).encode()
            + b',"\\\\ud800"]}]}'
        ),
    )
