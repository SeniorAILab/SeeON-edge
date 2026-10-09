from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from backend.app.features.cameras import topology_client as topology_client_module
from backend.app.features.cameras.edge_topology_sync_state import TopologyPauseReason
from backend.app.features.cameras.topology_client import TopologyClient, TopologyPaused
from contracts.edge_provisioning_models import EnrollmentVerificationResult, FacilityIdentity
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologyConfirmation

SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
TOKEN = "server-side-secret-token"
EVENTS_URL = "https://product.example/api/v1/events"
PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3)
FACILITY_CODE = "NH-7H2K9M4QXP"
INSTALLATION_REF = "aa83ea3f-6e5f-4f45-a401-fb36c38835b6"
FACILITY = FacilityIdentity("87d79f24-b32f-49a3-b534-19f0af7d9135", "Ward A")


def _unused_enrollment_check(
    events_url: str,
    facility_code: str,
    client_installation_ref: str,
    facility_token: str,
    timeout_sec: float,
) -> EnrollmentVerificationResult | None:
    raise AssertionError("enrollment must not be checked")


def _client(check: Callable[..., EnrollmentVerificationResult | None]) -> TopologyClient:
    return TopologyClient(EVENTS_URL, TOKEN, PRINCIPAL, FACILITY_CODE, INSTALLATION_REF, 1.0, check)


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (401, TopologyPauseReason.AUTH),
        (403, TopologyPauseReason.FORBIDDEN),
        (409, TopologyPauseReason.CONFLICT),
    ],
)
def test_confirmation_classifies_upstream_auth_and_conflict_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    reason: TopologyPauseReason,
) -> None:
    captured: list[tuple[str, str, dict[str, str], bytes]] = []

    def request(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes,
        _timeout: float,
    ) -> tuple[int, dict[str, str], bytes]:
        captured.append((url, method, headers, body))
        return status_code, {}, b"{}"

    monkeypatch.setattr(topology_client_module, "bounded_request", request)
    client = _client(_unused_enrollment_check)

    result = client.confirm(SNAPSHOT_ID, TopologyConfirmation(CONFIRMATION_ID, "a" * 64, 7))

    assert result == TopologyPaused(reason, status_code)
    assert captured == [
        (
            f"https://product.example/api/v1/edge/topology-snapshots/{SNAPSHOT_ID}/confirm",
            "POST",
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            json.dumps(
                {
                    "confirmationId": CONFIRMATION_ID,
                    "digest": "a" * 64,
                    "expectedServerRevision": 7,
                    "schemaVersion": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
        )
    ]


@pytest.mark.parametrize(
    ("verified", "expected"),
    [
        (EnrollmentVerificationResult(PRINCIPAL, FACILITY, 9), 9),
        (
            EnrollmentVerificationResult(
                MachinePrincipal(PRINCIPAL.edge_installation_id, 4), FACILITY, 9
            ),
            None,
        ),
        (None, None),
    ],
)
def test_server_revision_comes_from_the_injected_enrollment_check(
    verified: EnrollmentVerificationResult | None, expected: int | None
) -> None:
    calls: list[tuple[str, str, str, str, float]] = []

    def check(
        events_url: str,
        facility_code: str,
        client_installation_ref: str,
        facility_token: str,
        timeout_sec: float,
    ) -> EnrollmentVerificationResult | None:
        calls.append(
            (events_url, facility_code, client_installation_ref, facility_token, timeout_sec)
        )
        return verified

    assert _client(check).refresh_server_revision() == expected
    assert calls == [(EVENTS_URL, FACILITY_CODE, INSTALLATION_REF, TOKEN, 1.0)]
