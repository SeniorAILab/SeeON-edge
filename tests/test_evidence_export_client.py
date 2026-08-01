from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

import pytest

from shared.events.evidence_export_client import (
    MAX_LOCAL_EVENT_PAYLOAD_BYTES,
    BackendEvidenceClient,
    RelayEvidenceClient,
)
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.evidence_http_transport import MAX_RESPONSE_BYTES
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimedClip,
    ClipId,
    ClipLocalState,
    EdgeEventId,
)

EVENT_ID = EdgeEventId("00000000-0000-4000-8000-000000000001")


class ScriptedHandler(BaseHTTPRequestHandler):
    responses: ClassVar[list[tuple[int, dict[str, str], bytes]]] = []
    requests: ClassVar[list[tuple[str, str, bytes, str | None]]] = []

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def do_PUT(self) -> None:
        self._respond()

    def _respond(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).requests.append(
            (
                self.command,
                self.path,
                body,
                self.headers.get("X-Edge-Relay-Token"),
            )
        )
        status, headers, payload = type(self).responses.pop(0)
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: str | int) -> None:
        return


@pytest.fixture
def server_url():
    ScriptedHandler.responses = []
    ScriptedHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _json(payload: dict[str, str | int | None]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _claimed_clip() -> ClaimedClip:
    return ClaimedClip(
        clip_id=ClipId("clip-1"),
        local_state=ClipLocalState.VERIFIED,
        state_version=2,
        media_relpath="clips/clip-1/clip.mp4",
        sha256="a" * 64,
        size_bytes=4,
        mime_type="video/mp4",
        codec="h264",
        duration_ms=1000,
        clip_start_at="2026-07-16T00:00:00Z",
        clip_end_at="2026-07-16T00:00:01Z",
        finalized_at="2026-07-16T00:00:02Z",
        unavailable_reason=None,
        event_ids=(EVENT_ID,),
        event_payload_json='{"camera_id":"camera-1","facility_id":"facility-1"}',
        lease_owner="sender",
        lease_expires_at=10.0,
        attempt_count=1,
    )


def test_relay_client_uses_versioned_capabilities_route(server_url: str) -> None:
    ScriptedHandler.responses = [(200, {}, _json({"event_idempotency": 1, "clip_export": 1}))]
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)

    result = client.probe_capabilities("camera-1")

    assert result == BackendCapabilities(event_idempotency=1, clip_export=1)
    assert [(method, path, token) for method, path, _body, token in ScriptedHandler.requests] == [
        ("GET", "/api/v1/relay/capabilities?camera_id=camera-1", "relay-secret")
    ]


def test_relay_client_parses_typed_event_receipt_and_retry_after(server_url: str) -> None:
    # Given: a relay returns one accepted event and one overloaded response.
    ScriptedHandler.responses = [
        (202, {}, _json({"status": "accepted", "edge_event_id": EVENT_ID, "event_id": "evt"})),
        (429, {"retry-after": "45"}, b'{"detail":"redacted by client"}'),
    ]
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)
    payload = _json({"edge_event_id": EVENT_ID}).decode()

    # When: both outcomes cross the HTTP boundary.
    receipt = client.send_event(payload, EVENT_ID)
    retry = client.send_event(payload, EVENT_ID)

    # Then: only typed/sanitized results escape the client.
    assert receipt == EventReceipt("accepted", EVENT_ID, "evt")
    assert retry == DeliveryFailure(
        DeliveryDisposition.RETRY,
        "HTTP_429",
        status_code=429,
        retry_after_seconds=45.0,
    )
    assert "redacted by client" not in repr(retry)
    assert [(method, path, token) for method, path, _body, token in ScriptedHandler.requests] == [
        ("POST", "/api/v1/relay/alerts", "relay-secret"),
        ("POST", "/api/v1/relay/alerts", "relay-secret"),
    ]


def test_relay_client_treats_exact_409_receipt_as_idempotent_success(server_url: str) -> None:
    # Given: response loss caused an exact immutable READY replay.
    ScriptedHandler.responses = [
        (
            409,
            {},
            _json(
                {
                    "clip_id": "clip-1",
                    "state": "READY",
                    "state_version": 2,
                    "sha256": "a" * 64,
                    "size_bytes": 4,
                }
            ),
        )
    ]
    media = Path("/dev/null")
    claim = ClaimedClip(
        clip_id=ClipId("clip-1"),
        local_state=ClipLocalState.VERIFIED,
        state_version=2,
        media_relpath=str(media),
        sha256="a" * 64,
        size_bytes=4,
        mime_type="video/mp4",
        codec="h264",
        duration_ms=1000,
        clip_start_at="2026-07-16T00:00:00Z",
        clip_end_at="2026-07-16T00:00:01Z",
        finalized_at="2026-07-16T00:00:02Z",
        unavailable_reason=None,
        event_ids=(EVENT_ID,),
        event_payload_json='{"camera_id":"camera-1","facility_id":"facility-1"}',
        lease_owner="sender",
        lease_expires_at=10.0,
        attempt_count=1,
    )
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)

    # When: the relay returns the original receipt with 409.
    result = client.send_clip(claim)

    # Then: the exact receipt converges as success, not dead-letter.
    assert result == ClipReceipt("clip-1", "READY", 2, "a" * 64, 4)
    assert [(method, path, token) for method, path, _body, token in ScriptedHandler.requests] == [
        ("PUT", "/api/v1/relay/clips/clip-1", "relay-secret")
    ]


@pytest.mark.parametrize("status", (404, 405))
def test_backend_capability_old_endpoint_is_compatibility_block(
    server_url: str,
    status: int,
) -> None:
    # Given: an old backend does not implement event media capability.
    ScriptedHandler.responses = [(status, {}, b"")]
    client = BackendEvidenceClient(f"{server_url}/api/v1/events", "backend-secret", timeout_sec=1.0)

    # When: ml-api probes the backend.
    result = client.probe_capabilities("camera-1")

    # Then: it is a durable compatibility block, not a retry storm.
    assert result == DeliveryFailure(
        DeliveryDisposition.COMPATIBILITY,
        f"HTTP_{status}",
        status_code=status,
    )


@pytest.mark.parametrize("url", ("file:///tmp/events", "ftp://example.test/events", "/relative"))
def test_evidence_clients_reject_non_http_urls(url: str) -> None:
    with pytest.raises(ValueError, match="HTTP"):
        RelayEvidenceClient(url, "secret")
    with pytest.raises(ValueError, match="HTTP"):
        BackendEvidenceClient(url, "secret")


def test_relay_client_accepts_terminal_expired_receipt(server_url: str) -> None:
    ScriptedHandler.responses = [
        (
            200,
            {},
            _json(
                {
                    "clip_id": "clip-1",
                    "state": "EXPIRED",
                    "state_version": 3,
                    "sha256": None,
                    "size_bytes": None,
                }
            ),
        )
    ]
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)
    result = client.send_clip(_claimed_clip())
    assert result == ClipReceipt("clip-1", "EXPIRED", 3, None, None)


def test_relay_client_preserves_identity_from_large_durable_event(
    server_url: str,
) -> None:
    # Given: one valid durable event is larger than the bounded response contract.
    event_payload_json = json.dumps(
        {
            "camera_id": "camera-large",
            "facility_id": "facility-large",
            "snapshot_jpeg_base64": "x" * (64 * 1024),
        },
        separators=(",", ":"),
    )
    claim = replace(_claimed_clip(), event_payload_json=event_payload_json)
    ScriptedHandler.responses = [
        (
            200,
            {},
            _json(
                {
                    "clip_id": "clip-1",
                    "state": "READY",
                    "state_version": 2,
                    "sha256": "a" * 64,
                    "size_bytes": 4,
                }
            ),
        )
    ]
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)

    # When: the actual relay client constructs and sends the clip request.
    result = client.send_clip(claim)

    # Then: local payload size does not erase the required identity fields.
    request_payload = cast(dict[str, object], json.loads(ScriptedHandler.requests[0][2]))
    assert len(event_payload_json.encode()) > MAX_RESPONSE_BYTES
    assert result == ClipReceipt("clip-1", "READY", 2, "a" * 64, 4)
    assert request_payload["camera_id"] == "camera-large"
    assert request_payload["facility_id"] == "facility-large"


@pytest.mark.parametrize(
    "event_payload_json",
    (
        "{broken",
        '{"camera_id":"camera-1"}',
        json.dumps(
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "snapshot_jpeg_base64": "x" * MAX_LOCAL_EVENT_PAYLOAD_BYTES,
            },
            separators=(",", ":"),
        ),
    ),
)
def test_relay_client_rejects_invalid_local_event_identity_without_http(
    server_url: str,
    event_payload_json: str,
) -> None:
    client = RelayEvidenceClient(server_url, "relay-secret", timeout_sec=1.0)
    claim = replace(_claimed_clip(), event_payload_json=event_payload_json)

    result = client.send_clip(claim)

    assert result == DeliveryFailure(
        DeliveryDisposition.PERMANENT,
        "INVALID_EVENT_PAYLOAD",
    )
    assert ScriptedHandler.requests == []
