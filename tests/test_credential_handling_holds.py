import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.app.features.connection.enrollment import (
    EnrollmentCredentials,
    EnrollmentVerificationFailure,
    verify_enrollment,
)
from backend.app.features.connection.hub_url import API_BACKEND_ALLOW_INSECURE_HTTP_ENV
from backend.app.features.relay.auth import authorize_relay
from backend.app.main import create_app, no_lifespan
from worker.types.trace import DecisionTraceSnapshot

CREDENTIALS = EnrollmentCredentials(
    facility_code="NH-7H2K9M4QXP",
    client_installation_ref="aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
    facility_token="facility-bearer-secret",
)
NON_ASCII_TOKEN = "relay-토큰-ñ"


def relay_request(token: str | None) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(edge_relay_token=token)))


def test_relay_token_comparison_accepts_a_matching_non_ascii_token() -> None:
    assert authorize_relay(relay_request(NON_ASCII_TOKEN), NON_ASCII_TOKEN) is None


def test_relay_token_comparison_rejects_a_non_ascii_mismatch_with_403_not_type_error() -> None:
    with pytest.raises(HTTPException) as raised:
        authorize_relay(relay_request(NON_ASCII_TOKEN), "relay-토큰-x")

    assert raised.value.status_code == 403


def test_relay_authority_is_the_app_state_token_never_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "seeded-at-startup")
    app = create_app(lifespan=no_lifespan)
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "changed-later")

    assert app.state.edge_relay_token == "seeded-at-startup"
    with pytest.raises(HTTPException) as raised:
        authorize_relay(SimpleNamespace(app=app), "changed-later")
    assert raised.value.status_code == 403
    assert authorize_relay(SimpleNamespace(app=app), "seeded-at-startup") is None


def test_enrollment_uses_the_default_tls_verification_and_sends_the_bearer_only_over_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, raising=False)
    calls: list[tuple[urllib.request.Request, dict[str, object]]] = []

    def refuse(request: urllib.request.Request, **kwargs: object) -> object:
        calls.append((request, kwargs))
        raise urllib.error.URLError(OSError("down"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    with pytest.raises(EnrollmentVerificationFailure) as raised:
        verify_enrollment("https://hub.example.com/api/v1/events", CREDENTIALS, timeout_sec=0.5)

    assert raised.value.error_class == "unreachable"
    [(request, kwargs)] = calls
    assert kwargs == {"timeout": 0.5}
    assert request.full_url == "https://hub.example.com/api/v1/edge/enrollments/verify"
    assert request.get_header("Authorization") == "Bearer facility-bearer-secret"


def test_enrollment_credentials_never_render_the_facility_token() -> None:
    assert "facility-bearer-secret" not in repr(CREDENTIALS)


@pytest.mark.parametrize("field", ["reason", "previous_state", "current_state"])
def test_rejected_trace_vocabulary_is_never_echoed_back(field: str) -> None:
    leaked = "Bearer sk-live-secret /home/operator/.ssh/id_rsa"
    fields = {
        "reason": "fall-onset",
        "previous_state": "clear",
        "current_state": "fall",
        field: leaked,
    }

    with pytest.raises(ValueError) as raised:
        DecisionTraceSnapshot(triggered=False, track_id=None, bed_id=None, **fields)

    assert leaked not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
