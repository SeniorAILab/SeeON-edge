from __future__ import annotations

import time
from typing import final

from contracts.decode_diagnostics import DecodeSelection
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RuntimeStatusSender,
    RuntimeStatusSenderConfig,
)
from worker.runtime.telemetry.wire import RelayRuntimeStatusPayload

# test_sender_uses_latest_snapshot_and_bearer_auth (edge): payload shape, generation
# bookkeeping, and bearer auth are superseded by tests/test_worker_telemetry_status_sender.py
# ::test_sender_preserves_generation_and_monotonic_sequence and
# ::test_relay_transport_uses_bounded_status_endpoint_and_parses_receipt (auth header lives
# at the RelayRuntimeStatusTransport boundary now). Neither covers start()/stop() actually
# delivering a queued snapshot through the background thread -- ported below.
# test_sender_posts_separate_facility_payloads_with_only_bound_cameras (edge): per-facility
# partitioning is superseded by
# tests/test_worker_telemetry_status_sender.py::test_sender_partitions_cameras_by_facility
# (exercised synchronously via publish_once()); the thread-lifecycle test below already drives
# the same _run() loop for the single-facility case, so re-porting the multi-facility split
# would only duplicate that loop, not add coverage.


@final
class _RecordingTransport:
    __slots__ = ("payloads",)

    def __init__(self) -> None:
        self.payloads: list[RelayRuntimeStatusPayload] = []

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None:
        self.payloads.append(payload)
        return 7


@final
class _FlakyTransport:
    __slots__ = ("attempts",)

    def __init__(self) -> None:
        self.attempts = 0

    def send(self, _payload: RelayRuntimeStatusPayload) -> int | None:
        self.attempts += 1
        return None if self.attempts == 1 else 3


@final
class _UnusedTransport:
    def send(self, _payload: RelayRuntimeStatusPayload) -> int | None:
        raise AssertionError("transport.send must not be called before start()")


def _diagnostics() -> WorkerDiagnostics:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-a",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=1,
            last_reason="spawn_failed",
            updated_at_sec=1.0,
        ),
    )
    return diagnostics


def test_sender_start_delivers_queued_snapshot_via_background_thread() -> None:
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        _diagnostics(),
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    sender.start()
    try:
        _wait_until(lambda: bool(transport.payloads))
    finally:
        sender.stop()

    assert transport.payloads[0]["cameras"] == [
        {
            "camera_id": "camera-a",
            "decode": {
                "requested": "auto",
                "selected": "nvdec",
                "fallback_count": 1,
                "last_reason": "spawn_failed",
                "updated_at_sec": 1.0,
            },
        }
    ]
    assert sender.generation == 7
    assert sender.is_alive is False


def test_sender_retries_with_backoff_and_recovers() -> None:
    transport = _FlakyTransport()
    sender = RuntimeStatusSender(
        _diagnostics(),
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(
            publish_interval_sec=1.0,
            initial_backoff_sec=0.01,
            max_backoff_sec=0.02,
        ),
    )

    sender.start()
    _wait_until(lambda: transport.attempts >= 2)
    sender.stop()

    assert sender.generation == 3
    assert sender.is_alive is False


def test_sender_publish_never_blocks_when_latest_slot_is_full() -> None:
    sender = RuntimeStatusSender(_diagnostics(), "facility-a", _UnusedTransport())

    assert sender.publish() is True
    start = time.monotonic()
    assert sender.publish() is True

    assert time.monotonic() - start < 0.1


def _wait_until(predicate, timeout_sec: float = 0.5) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for runtime status sender")
