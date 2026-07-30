from __future__ import annotations

import json
import time
import urllib.error
from dataclasses import dataclass

from contracts.decode_diagnostics import DecodeSelection
from edge.runtime.runtime_diagnostics import WorkerDiagnostics
from edge.runtime.runtime_status_sender import RuntimeStatusSender


@dataclass
class _Response:
    payload: dict[str, object]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_sender_uses_latest_snapshot_and_bearer_auth() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection(selected="nvdec"))
    received: list[tuple[dict[str, object], str | None]] = []

    def urlopen(request, *, timeout: float):  # noqa: ANN001
        del timeout
        received.append(
            (
                json.loads(request.data.decode("utf-8")),
                request.get_header("Authorization"),
            )
        )
        return _Response({"accepted": True, "generation": 7})

    sender = RuntimeStatusSender(
        "http://relay.test/",
        "relay-token",
        diagnostics,
        "facility-a",
        publish_interval_sec=0.01,
        urlopen=urlopen,
    )
    sender.publish()
    diagnostics.update_decode("camera-a", _selection(selected="opencv"))
    sender.publish()
    sender.start()
    try:
        _wait_until(lambda: bool(received))
    finally:
        sender.stop()

    payload, authorization = received[0]
    assert authorization == "Bearer relay-token"
    assert payload["generation"] is None
    assert payload["seq"] == 1
    assert payload["cameras"] == [
        {
            "camera_id": "camera-a",
            "decode": {
                "requested": "auto",
                "selected": "opencv",
                "fallback_count": 1,
                "last_reason": "spawn_failed",
                "updated_at_sec": 1.0,
            },
        }
    ]
    assert "user:password" not in json.dumps(payload)
    assert sender.generation == 7
def test_sender_posts_separate_facility_payloads_with_only_bound_cameras() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection(selected="nvdec"))
    diagnostics.update_decode("camera-b", _selection(selected="opencv"))
    received: list[dict[str, object]] = []

    def urlopen(request, *, timeout: float):  # noqa: ANN001
        del timeout
        received.append(json.loads(request.data.decode("utf-8")))
        return _Response({"accepted": True, "generation": 7})

    sender = RuntimeStatusSender(
        "http://relay.test",
        "relay-token",
        diagnostics,
        {"camera-a": "facility-a", "camera-b": "facility-b"},
        publish_interval_sec=1.0,
        urlopen=urlopen,
    )
    sender.start()
    try:
        _wait_until(lambda: len(received) >= 2)
    finally:
        sender.stop()

    payloads = {str(payload["facility_id"]): payload for payload in received[:2]}
    assert [
        camera["camera_id"] for camera in payloads["facility-a"]["cameras"]
    ] == ["camera-a"]
    assert [
        camera["camera_id"] for camera in payloads["facility-b"]["cameras"]
    ] == ["camera-b"]



def test_sender_retries_with_backoff_and_stops_cleanly() -> None:
    diagnostics = WorkerDiagnostics()
    calls = 0

    def urlopen(request, *, timeout: float):  # noqa: ANN001
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("unavailable")
        return _Response({"accepted": True, "generation": 3})

    sender = RuntimeStatusSender(
        "http://relay.test",
        "relay-token",
        diagnostics,
        "facility-a",
        publish_interval_sec=1.0,
        initial_backoff_sec=0.01,
        max_backoff_sec=0.02,
        urlopen=urlopen,
    )
    sender.start()
    _wait_until(lambda: calls >= 2)
    sender.stop()

    assert sender.generation == 3
    assert sender.is_alive is False


def test_sender_publish_never_blocks_when_latest_slot_is_full() -> None:
    diagnostics = WorkerDiagnostics()
    sender = RuntimeStatusSender(
        "http://relay.test",
        "relay-token",
        diagnostics,
        "facility-a",
    )

    assert sender.publish() is True
    start = time.monotonic()
    assert sender.publish() is True

    assert time.monotonic() - start < 0.1


def _selection(*, selected: str) -> DecodeSelection:
    return DecodeSelection(
        requested="auto",
        selected=selected,
        fallback_count=1,
        last_reason="spawn_failed",
        updated_at_sec=1.0,
    )


def _wait_until(predicate, timeout_sec: float = 1.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for runtime status sender")
