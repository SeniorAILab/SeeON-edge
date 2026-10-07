from __future__ import annotations

import pytest

from shared.events.evidence_export_contract import DeliveryDisposition
from shared.events.evidence_http_transport import classify_http_failure


@pytest.mark.parametrize("status", (401, 403))
def test_ambient_auth_failures_are_retried_not_dead_lettered(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.code == f"HTTP_{status}"
    assert failure.status_code == status


@pytest.mark.parametrize("status", (400, 413, 415, 422))
def test_payload_specific_failures_remain_permanent(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.PERMANENT
    assert failure.code == f"HTTP_{status}"


@pytest.mark.parametrize("status", (404, 405))
def test_compatibility_classes_are_unaffected(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.COMPATIBILITY


@pytest.mark.parametrize("status", (408, 425, 429, 500, 503, 599))
def test_transient_classes_remain_retry(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.RETRY


def test_retry_after_header_is_preserved_for_ambient_auth_failures() -> None:
    failure = classify_http_failure(403, {"Retry-After": "30"})

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.retry_after_seconds == 30.0


def test_named_local_accept_is_terminal_and_absent_status_is_not() -> None:
    from shared.events.evidence_export_contract import DeliveryFailure, EventReceipt
    from shared.events.evidence_http_transport import parse_event_result

    named = parse_event_result(
        (202, {}, b'{"status": "accepted_local", "edge_event_id": "edge-1"}'), "edge-1"
    )
    assert isinstance(named, EventReceipt), (
        "a named local accept must be terminal; treating it as a failure is the "
        "defect that wedged the queue"
    )
    assert named.status == "accepted_local"
    assert named.edge_event_id == "edge-1"
    assert named.event_id == "", "a local accept has no upstream id to fabricate"

    bare = parse_event_result((202, {}, b'{"status": "accepted"}'), "edge-1")
    assert isinstance(bare, DeliveryFailure), "an unnamed 2xx is still malformed"
    assert bare.code == "MALFORMED_RECEIPT"

    wrong = parse_event_result(
        (202, {}, b'{"status": "accepted_local", "edge_event_id": "other"}'), "edge-1"
    )
    assert isinstance(wrong, DeliveryFailure), (
        "a local accept naming another event must not acknowledge this one"
    )


def test_a_terminal_local_accept_requires_that_something_was_persisted() -> None:
    from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
    from shared.events.evidence_http_transport import parse_event_result

    refused = parse_event_result(
        (503, {}, b'{"detail": "edge-local persistence failed"}'), "edge-1"
    )
    assert isinstance(refused, DeliveryFailure)
    assert refused.disposition is DeliveryDisposition.RETRY, (
        "a backend that could not persist the alert must not cause the worker "
        "to drop it; the event would then exist nowhere"
    )
