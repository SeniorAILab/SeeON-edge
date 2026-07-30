from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Self

import pytest
from edge_worker_fixtures import edge_config_payload
from pydantic import ValidationError

from backend.app.features.relay.router import RelayAlertRequest
from contracts.relay import EventApiPayload
from edge.runtime.edge_worker_config import EdgeWorkerConfig
from shared.events.edge_ingest_client import EdgeIngestClient


def test_edge_worker_config_uses_local_relay_and_has_no_backend_ingest_urls() -> None:
    config = EdgeWorkerConfig.model_validate(edge_config_payload(camera_count=1))

    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert not hasattr(config, "alert_api_url")
    assert not hasattr(config, "heartbeat_api_url")


@pytest.mark.parametrize("field_name", ["ingest", "alert_api_url", "heartbeat_api_url"])
def test_edge_worker_config_rejects_backend_ingest_fields(field_name: str) -> None:
    payload = edge_config_payload(camera_count=1)
    payload[field_name] = (
        {"alert_api_url": "http://backend.local/ingest/alerts"}
        if field_name == "ingest"
        else "http://backend.local/ingest/alerts"
    )

    with pytest.raises(ValidationError, match=field_name):
        EdgeWorkerConfig.model_validate(payload)


@pytest.mark.parametrize("field_name", ["ingest_" + "key_id", "ingest_" + "secret"])
def test_edge_worker_config_rejects_camera_backend_credentials(field_name: str) -> None:
    payload = edge_config_payload(camera_count=1)
    payload["cameras"][0][field_name] = "backend-secret"

    with pytest.raises(ValidationError, match=field_name):
        EdgeWorkerConfig.model_validate(payload)


def test_worker_relay_client_posts_to_local_relay_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edge.runtime import edge_worker

    captured: list[tuple[str, dict[str, str], dict[str, object]]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 0.5
        captured.append(
            (
                request.full_url,
                dict(request.header_items()),
                json.loads(request.data.decode("utf-8")),
            )
        )
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = EdgeWorkerConfig.model_validate(edge_config_payload(camera_count=1))
    camera = config.cameras[0]
    client = edge_worker._relay_client(config, camera)

    client.emit(
        {
            "event_type": "bed-exit",
            "detected_at": "2026-06-23T12:00:00.000Z",
            "probability": 0.91,
        }
    )
    client.send_heartbeat()

    assert captured == [
        (
            "http://127.0.0.1:8000/api/v1/relay/alerts",
            {"Content-type": "application/json", "X-edge-relay-token": "relay-token-1"},
            {
                "event_type": "bed-exit",
                "probability": 0.91,
                "detected_at": "2026-06-23T12:00:00.000Z",
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "evidence": {
                    "event_type": "bed-exit",
                    "detected_at": "2026-06-23T12:00:00.000Z",
                    "probability": 0.91,
                },
                "resident_id": "resident-1",
            },
        ),
        (
            "http://127.0.0.1:8000/api/v1/relay/heartbeat",
            {"Content-type": "application/json", "X-edge-relay-token": "relay-token-1"},
            {"camera_id": "camera-1", "facility_id": "facility-1", "config_version": 0},
        ),
    ]



def test_relay_alert_request_accepts_optional_audit_and_snapshot() -> None:
    envelope_less = RelayAlertRequest.model_validate(
        {
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": "2026-06-23T12:00:00.000Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
        }
    )
    envelope_bearing = RelayAlertRequest.model_validate(
        {
            "event_type": "bed-exit",
            "probability": 0.87,
            "detected_at": "2026-06-23T12:01:00.000Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "audit": {
                "config_version": 3,
                "model_version": "model-v1",
                "detector_version": "detector-v1",
                "operating_threshold": 0.72,
                "clock_source": "edge_wall_clock",
            },
            "snapshot_jpeg_base64": "anBlZw==",
        }
    )

    assert envelope_less.audit is None
    assert envelope_less.snapshot_jpeg_base64 is None
    assert envelope_bearing.audit is not None
    assert envelope_bearing.audit.model_dump(exclude_none=True) == {
        "config_version": 3,
        "model_version": "model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.72,
        "clock_source": "edge_wall_clock",
    }
    assert envelope_bearing.snapshot_jpeg_base64 == "anBlZw=="

    with pytest.raises(ValidationError, match="unexpected"):
        RelayAlertRequest.model_validate(
            {
                "event_type": "fall",
                "probability": 0.91,
                "detected_at": "2026-06-23T12:00:00.000Z",
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "unexpected": "field",
            }
        )


def test_event_api_payload_omits_audit_when_unset_and_includes_when_set() -> None:
    envelope_less = EventApiPayload(
        camera_id="camera-1",
        type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        confidence=0.91,
    ).as_dict()
    envelope_bearing = EventApiPayload(
        camera_id="camera-1",
        type="bed-exit",
        detected_at="2026-06-23T12:01:00.000Z",
        confidence=0.87,
        config_version=3,
        model_version="model-v1",
        detector_version="detector-v1",
        operating_threshold=0.72,
        clock_source="edge_wall_clock",
    ).as_dict()

    assert envelope_less == {
        "camera_id": "camera-1",
        "type": "fall",
        "detected_at": "2026-06-23T12:00:00.000Z",
        "confidence": 0.91,
    }
    assert _backend_policy_fields().isdisjoint(envelope_less)
    assert envelope_bearing == {
        "camera_id": "camera-1",
        "type": "bed-exit",
        "detected_at": "2026-06-23T12:01:00.000Z",
        "confidence": 0.87,
        "config_version": 3,
        "model_version": "model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.72,
        "clock_source": "edge_wall_clock",
    }


def test_edge_ingest_client_alert_payload_is_fact_event_shaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str | float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 0.2
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = EdgeIngestClient(
        events_url="http://backend.local/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )

    sent = client.send_alert(
        event_type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        probability=0.91,
    )

    assert sent is True
    assert captured_payloads == [
        {
            "camera_id": "camera-1",
            "type": "fall",
            "detected_at": "2026-06-23T12:00:00.000Z",
            "confidence": 0.91,
        }
    ]
    assert _backend_policy_fields().isdisjoint(captured_payloads[0])


def test_edge_ingest_client_uploads_snapshot_after_event_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str, bytes]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 0.2
        captured.append((request.get_method(), request.full_url, request.data))
        return _FakeHTTPResponse(b'{"id":"evt_1","status":"created"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = EdgeIngestClient(
        events_url="http://backend.local/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )

    sent = client.send_alert(
        event_type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        probability=0.91,
        snapshot_bytes=b"jpeg",
    )

    assert sent is True
    assert [method for method, _, _ in captured] == ["POST", "PUT"]
    assert captured[1] == ("PUT", "http://backend.local/events/evt_1/snapshot", b"jpeg")


def test_edge_ingest_client_without_snapshot_does_not_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_methods: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 0.2
        captured_methods.append(request.get_method())
        return _FakeHTTPResponse(b'{"id":"evt_1","status":"created"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = EdgeIngestClient(
        events_url="http://backend.local/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )

    sent = client.send_alert(
        event_type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        probability=0.91,
    )

    assert sent is True
    assert captured_methods == ["POST"]


def test_edge_ingest_client_snapshot_put_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_methods: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 0.2
        captured_methods.append(request.get_method())
        if request.get_method() == "PUT":
            raise urllib.error.URLError("snapshot down")
        return _FakeHTTPResponse(b'{"id":"evt_1","status":"created"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = EdgeIngestClient(
        events_url="http://backend.local/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )

    sent = client.send_alert(
        event_type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        probability=0.91,
        snapshot_bytes=b"jpeg",
    )

    assert sent is True
    assert captured_methods == ["POST", "PUT"]



class _FakeHTTPResponse:
    def __init__(self, body: bytes = b"") -> None:
        self._body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _backend_policy_fields() -> set[str]:
    return {
        "notification_recipient",
        "notification_channel",
        "recipient",
        "channel",
        "dedup_key",
        "deduplication_key",
        "outbox_id",
        "email_template",
        "email_delivery_id",
    }
