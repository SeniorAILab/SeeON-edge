from __future__ import annotations

from typing import Any

import pytest

from backend.app.main import create_app, no_lifespan
from shared.events.execution_records import RECORD_KINDS, TIME_QUALITIES

_PATH = "/api/v1/relay/execution-records"


@pytest.fixture(scope="module")
def openapi() -> dict[str, Any]:
    return create_app(lifespan=no_lifespan).openapi()


def _schema(openapi: dict[str, Any], name: str) -> dict[str, Any]:
    return openapi["components"]["schemas"][name]


def test_ingest_operation_publishes_a_required_json_request_body(openapi: dict[str, Any]) -> None:
    body = openapi["paths"][_PATH]["post"]["requestBody"]
    assert body["required"] is True
    assert body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ExecutionRecordBatchRequest"
    }


def test_request_schema_requires_the_batch_envelope(openapi: dict[str, Any]) -> None:
    batch = _schema(openapi, "ExecutionRecordBatchRequest")
    assert set(batch["required"]) == {"camera_id", "worker_boot_id", "provenance"}
    assert set(batch["properties"]) == {
        "camera_id",
        "worker_boot_id",
        "provenance",
        "records",
        "gaps",
        "batch_id",
    }


def test_request_schema_vocabulary_matches_the_wire_contract(openapi: dict[str, Any]) -> None:
    record = _schema(openapi, "ExecutionRecordRequest")
    assert set(record["properties"]["record_kind"]["enum"]) == RECORD_KINDS
    assert set(record["properties"]["time_quality"]["enum"]) == TIME_QUALITIES
    assert record["additionalProperties"] is False
