from __future__ import annotations

from typing import Any, Final

import pytest

from backend.app.main import create_app, no_lifespan

ROUTE_DESCRIPTIONS: Final = {
    "/api/v1/relay/alerts": (
        "Commit the incident and its delivery obligation, then answer.\n"
        "\n"
        "Every response is built after the admission COMMIT, so a failure before it\n"
        "is never acknowledged. A worker retry after a lost response lands on the\n"
        "same committed row instead of creating a second incident."
    ),
    "/api/v1/relay/snapshot-attachments": (
        "Record one immutable media reference without accepting media bytes."
    ),
    "/api/v1/relay/snapshot-dispositions": (
        "Durably record an unavailable or failed snapshot without touching its event."
    ),
}

SCHEMA_DESCRIPTIONS: Final = {
    "RelaySnapshotAttachmentRequest": (
        "An immutable snapshot reference; snapshot bytes never cross this route."
    ),
    "RelaySnapshotDispositionRequest": (
        "A terminal, explicit statement that a snapshot cannot be delivered."
    ),
}


@pytest.fixture(scope="module")
def openapi() -> dict[str, Any]:
    return create_app(lifespan=no_lifespan).openapi()


@pytest.mark.parametrize(("path", "expected"), sorted(ROUTE_DESCRIPTIONS.items()))
def test_relay_route_publishes_its_openapi_description(
    openapi: dict[str, Any], path: str, expected: str
) -> None:
    assert expected
    assert openapi["paths"][path]["post"]["description"] == expected


@pytest.mark.parametrize(("schema", "expected"), sorted(SCHEMA_DESCRIPTIONS.items()))
def test_relay_request_schema_publishes_its_openapi_description(
    openapi: dict[str, Any], schema: str, expected: str
) -> None:
    assert expected
    assert openapi["components"]["schemas"][schema]["description"] == expected
