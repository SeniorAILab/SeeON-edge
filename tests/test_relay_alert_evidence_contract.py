from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api_ingest_relay import FakeBackendIngestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.relay.router import RelayAlertRequest
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import parse_event_result
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_HEADERS, relay_postgres_app
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_sender import _payload
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.types.business_event import BusinessEvent

pytest_plugins = ("tests_support.postgres_sandbox",)

_PATH = "/api/v1/relay/alerts"
_HEADERS = {**RELAY_HEADERS, "Content-Type": "application/json"}
_EVENT_ID = "00000000-0000-4000-9000-000000000001"

WORKER_SHAPES: list[tuple[str, dict[str, Any]]] = [
    ("fall", {}),
    ("bed-exit", {"domain": "bed_exit", "event_type": "bed-exit", "bed_id": 2}),
    ("detection-lost", {"domain": "detection-lost", "event_type": "detection-lost"}),
    ("no-person-no-bed", {"person_id": None}),
    ("zero-values", {"person_id": 0, "bed_id": 0, "time_sec": 0.0, "probability": 0.0}),
    ("integer-time", {"time_sec": 30, "probability": 1.0}),
    ("unicode-domain", {"domain": "낙상-module.v2"}),
    ("large-ids", {"person_id": 2**31, "bed_id": 2**31 + 1, "time_sec": 1.7e9 + 0.000123}),
]

ACCEPTED: list[tuple[str, object]] = [
    ("domain-and-clip", {"domain": "night-bed-exit", "clip_id": "clip-123"}),
    ("large-detail", {"detail": "x" * 1024}),
    ("deep", {"child": {"child": {"child": {}}}}),
    ("empty", {}),
    ("known-nulls", {"domain": None, "identity": None, "time_sec": None, "person_id": None}),
    ("extra-nulls", {"domain": "fall", "x": None, "n": {"a": None}}),
    ("integer-identity", {"identity": 42}),
    ("integer-time", {"time_sec": 30}),
    ("whole-float-time", {"time_sec": 30.0}),
    ("unicode-keys", {"도메인": "낙상"}),
    ("model-attribute-keys", {"model_config": 1, "json": 2, "copy": 3, "__class__": 4}),
]

REJECTED: list[tuple[str, object, list[str | int]]] = [
    ("domain-number", {"domain": 5}, ["body", "evidence", "domain"]),
    ("identity-float", {"identity": 1.5}, ["body", "evidence", "identity", "str"]),
    ("time-text", {"time_sec": "12.5"}, ["body", "evidence", "time_sec", "int"]),
    ("time-bool", {"time_sec": True}, ["body", "evidence", "time_sec", "int"]),
    ("person-text", {"person_id": "7"}, ["body", "evidence", "person_id"]),
    ("person-float", {"person_id": 7.0}, ["body", "evidence", "person_id"]),
    ("bed-bool", {"bed_id": True}, ["body", "evidence", "bed_id"]),
    ("clip-number", {"clip_id": 5}, ["body", "evidence", "clip_id"]),
    ("evidence-list", [1], ["body", "evidence"]),
    ("evidence-text", "fall", ["body", "evidence"]),
]

UNENCODABLE: list[tuple[str, object, str]] = [
    ("surrogate-domain", {"domain": "\ud800"}, "surrogates not allowed"),
    ("surrogate-key", {"\ud800": 1}, "surrogates not allowed"),
    ("surrogate-extra", {"x": ["\ud800"]}, "surrogates not allowed"),
    ("surrogate-person", {"person_id": "\ud800"}, "surrogates not allowed"),
    ("nan-person", {"person_id": float("nan")}, "Out of range float values are not JSON compliant"),
    ("inf-time", {"time_sec": float("inf")}, "Out of range float values are not JSON compliant"),
    ("nan-domain", {"domain": float("nan")}, "Out of range float values are not JSON compliant"),
]


@pytest.fixture
def relay(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> TestClient:
    app = relay_postgres_app(
        postgres_product_sandbox, postgres_audit_runtime, client=FakeBackendIngestClient()
    )
    return TestClient(app, raise_server_exceptions=False)


class _NoClip:
    def on_event(self, trigger_packet: object, event: object, **_: object) -> None:
        return None


def _worker_body(tmp_path: Path, overrides: dict[str, Any]) -> bytes:
    fields: dict[str, Any] = {
        "domain": "fall",
        "event_type": "fall",
        "identity": _EVENT_ID,
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "time_sec": 12.5,
        "probability": 0.91,
        "person_id": 7,
        "bed_id": None,
        **overrides,
    }
    queue = tmp_path / "queue"
    stager = DurableEvidenceStager(
        queue_directory=queue, camera_id="camera-1", facility_id="facility-1", config_version=7
    )
    sink = EvidenceEventSink(
        stager=stager, recorder=_NoClip(), now=lambda: datetime(2026, 10, 9, 2, tzinfo=UTC)
    )
    sink.emit_for_frame(BusinessEvent(**fields), SimpleNamespace(camera_id="camera-1"))
    entries = [json.loads(path.read_text()) for path in sorted(queue.glob("*.json"))]
    (event,) = [entry for entry in entries if entry["kind"] == "EVENT"]
    return _payload(event).encode("utf-8")


def _body(evidence: object, **fields: object) -> bytes:
    body = {
        "edge_event_id": _EVENT_ID,
        "event_type": "fall",
        "probability": 0.5,
        "detected_at": "2026-10-09T02:00:00.000000Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "evidence": evidence,
        **fields,
    }
    return json.dumps(body, ensure_ascii=True, allow_nan=True).encode("ascii")


def _dto_accepts(body: bytes) -> bool:
    try:
        RelayAlertRequest.model_validate(json.loads(body))
    except ValidationError:
        return False
    return True


def _stored_evidence(sandbox: ProductSandbox, edge_event_id: str) -> str:
    row = sandbox.admin.execute(
        "SELECT envelope FROM event_outbox WHERE edge_event_id=%s", (edge_event_id,)
    ).fetchone()
    assert row is not None
    return json.dumps(json.loads(row[0])["evidence"], sort_keys=True)


def _disposition(response: Any) -> DeliveryDisposition | None:
    classified = parse_event_result((response.status_code, {}, response.content), _EVENT_ID)
    return classified.disposition if isinstance(classified, DeliveryFailure) else None


@pytest.mark.parametrize(("name", "overrides"), WORKER_SHAPES, ids=[n for n, _ in WORKER_SHAPES])
def test_every_worker_alert_shape_passes_the_dto_and_is_stored_verbatim(
    relay: TestClient,
    postgres_product_sandbox: ProductSandbox,
    tmp_path: Path,
    name: str,
    overrides: dict[str, Any],
) -> None:
    body = _worker_body(tmp_path, overrides)

    response = relay.post(_PATH, content=body, headers=_HEADERS)

    assert (_dto_accepts(body), response.status_code) == (True, 202), name
    sent = json.dumps(json.loads(body)["evidence"], sort_keys=True)
    assert _stored_evidence(postgres_product_sandbox, _EVENT_ID) == sent


@pytest.mark.parametrize(("name", "evidence"), ACCEPTED, ids=[n for n, _ in ACCEPTED])
def test_dto_and_endpoint_accept_the_same_evidence(
    relay: TestClient, postgres_product_sandbox: ProductSandbox, name: str, evidence: object
) -> None:
    body = _body(evidence)

    response = relay.post(_PATH, content=body, headers=_HEADERS)

    assert (_dto_accepts(body), response.status_code) == (True, 202), name
    assert _stored_evidence(postgres_product_sandbox, _EVENT_ID) == json.dumps(
        evidence, sort_keys=True
    )


@pytest.mark.parametrize(("name", "evidence", "loc"), REJECTED, ids=[n for n, _, _ in REJECTED])
def test_dto_and_endpoint_reject_the_same_evidence(
    relay: TestClient, name: str, evidence: object, loc: list[str | int]
) -> None:
    body = _body(evidence)

    response = relay.post(_PATH, content=body, headers=_HEADERS)

    assert (_dto_accepts(body), response.status_code) == (False, 422), name
    assert response.json()["detail"][0]["loc"] == loc
    assert _disposition(response) is DeliveryDisposition.PERMANENT


@pytest.mark.parametrize(
    ("name", "evidence", "detail"), UNENCODABLE, ids=[n for n, _, _ in UNENCODABLE]
)
def test_unencodable_evidence_is_refused_by_the_envelope_never_a_500(
    relay: TestClient, name: str, evidence: object, detail: str
) -> None:
    response = relay.post(_PATH, content=_body(evidence), headers=_HEADERS)

    assert response.status_code == 422, name
    assert detail in response.json()["detail"]
    assert _disposition(response) is DeliveryDisposition.PERMANENT


@pytest.mark.parametrize(
    ("evidence", "edge_event_id"),
    [
        ({"domain": "fall", "x": None, "time_sec": 30}, "d6764c42-4fdf-5cd6-9d04-108c010dbb59"),
        ({"time_sec": 30.0, "person_id": None}, "ff5afab6-fb31-5c76-a4a8-8262f03be756"),
        (None, "2f98960c-b1c2-50aa-baec-374583ebae93"),
    ],
)
def test_idless_alert_identity_still_hashes_the_evidence_as_sent(
    relay: TestClient,
    postgres_product_sandbox: ProductSandbox,
    evidence: object,
    edge_event_id: str,
) -> None:
    body = json.loads(_body(evidence))
    del body["edge_event_id"]

    response = relay.post(_PATH, json=body, headers=RELAY_HEADERS)

    assert response.status_code == 202
    rows = postgres_product_sandbox.admin.execute(
        "SELECT edge_event_id FROM event_outbox"
    ).fetchall()
    assert rows == [(edge_event_id,)]


def test_openapi_publishes_the_alert_evidence_schema() -> None:
    schemas = create_app(lifespan=no_lifespan).openapi()["components"]["schemas"]

    assert schemas["RelayAlertRequest"]["properties"]["evidence"]["anyOf"] == [
        {"$ref": "#/components/schemas/RelayAlertEvidence"},
        {"type": "null"},
    ]
    evidence = schemas["RelayAlertEvidence"]
    assert evidence["additionalProperties"] is True
    assert {
        name: sorted(option.get("type", "") for option in spec["anyOf"])
        for name, spec in evidence["properties"].items()
    } == {
        "domain": ["null", "string"],
        "identity": ["integer", "null", "string"],
        "time_sec": ["integer", "null", "number"],
        "person_id": ["integer", "null"],
        "bed_id": ["integer", "null"],
        "clip_id": ["null", "string"],
    }
