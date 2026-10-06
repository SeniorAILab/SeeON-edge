import json
import threading
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from backend.app.features.evidence import outbox_dispatch
from backend.app.features.evidence.outbox_delivery import DeliveryOutcome
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.runtime import worker as worker_runtime
from worker.runtime.clip_analysis_lifecycle import retry_teardown
from worker.runtime.nvidia_bed_zone_recognizer import (
    BedZoneRecognizerUnavailableError,
    NvidiaBedZoneRecognizer,
)
from worker.runtime.worker import EvidenceDeliveryError, NativeHeartbeatLoop, WorkerRuntime

PROCESS = object()
ENVELOPE = json.dumps(
    {
        "edge_event_id": "edge-event-1",
        "event_type": "fall",
        "detected_at": "2026-10-06T00:00:00Z",
        "probability": 0.9,
    }
)


def raising(error: BaseException) -> Callable[..., None]:
    def call(*args: object, **kwargs: object) -> None:
        raise error

    return call


def answering(result: object) -> Callable[..., object]:
    def call(*args: object, **kwargs: object) -> object:
        return result

    return call


class RecordingDelivery:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.finished: list[tuple[DeliveryOutcome, dict[str, object]]] = []

    def finish(self, claim: object, outcome: DeliveryOutcome, **fields: object) -> None:
        self.finished.append((outcome, {k: v for k, v in fields.items() if v is not None}))
        if self.error is not None:
            raise self.error


class FlakyTerminate:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, process: object) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("teardown failed")


class CountingPump:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.reads = 0

    @property
    def processed_count(self) -> int:
        self.reads += 1
        return self.reads


class Reporter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.marked: list[str] = []
        self.after_mark: Callable[[], None] = lambda: None

    def mark_ready(self, camera_id: str) -> None:
        self.marked.append(camera_id)
        self.after_mark()
        if self.error is not None:
            raise self.error


class RefusingServingClient:
    def create(self, *args: object, **kwargs: object) -> object:
        raise ConnectionError("serving endpoint down")


def claim() -> SimpleNamespace:
    return SimpleNamespace(
        backend_camera_id="camera-a", envelope=ENVELOPE, edge_event_id="edge-event-1"
    )


def send(result_or_error: object, delivery: RecordingDelivery) -> object:
    if isinstance(result_or_error, BaseException):
        client = SimpleNamespace(send_alert_receipt=raising(result_or_error))
    else:
        client = SimpleNamespace(send_alert_receipt=answering(result_or_error))
    return outbox_dispatch.dispatch(client, claim(), delivery)


def test_outbox_send_that_raises_is_recorded_as_unknown_and_kept_for_resend() -> None:
    delivery = RecordingDelivery()

    result = send(ConnectionError("reset"), delivery)

    assert result == DeliveryFailure(DeliveryDisposition.RETRY, "TRANSPORT_ERROR")
    assert delivery.finished == [(DeliveryOutcome.UNKNOWN, {"reason": "TRANSPORT_ERROR"})]


def test_outbox_keeps_the_obligation_when_the_hub_lacks_the_ingest_route() -> None:
    delivery = RecordingDelivery()

    send(DeliveryFailure(DeliveryDisposition.COMPATIBILITY, "NOT_FOUND", 404), delivery)

    assert delivery.finished == [
        (DeliveryOutcome.RETRY, {"reason": "NOT_FOUND", "http_status": 404})
    ]


def test_outbox_transport_failure_without_a_response_is_recorded_as_unknown() -> None:
    delivery = RecordingDelivery()
    failure = DeliveryFailure(
        DeliveryDisposition.RETRY, "TIMEOUT", transport_error="TimeoutError: timed out"
    )

    send(failure, delivery)

    assert [outcome for outcome, _ in delivery.finished] == [DeliveryOutcome.UNKNOWN]


def test_outbox_local_accept_is_terminal_and_carries_no_upstream_id() -> None:
    delivery = RecordingDelivery()

    send(EventReceipt("accepted_local", "edge-event-1", ""), delivery)

    assert delivery.finished == [(DeliveryOutcome.REJECTED, {"reason": "ACCEPTED_LOCAL"})]


def test_outbox_returns_the_hub_answer_even_when_recording_it_fails() -> None:
    receipt = EventReceipt("accepted", "edge-event-1", "hub-1")
    delivery = RecordingDelivery(error=ValueError("lease lost"))

    assert send(receipt, delivery) is receipt
    assert delivery.finished == [
        (DeliveryOutcome.SENT, {"reason": "ACCEPTED", "backend_event_id": "hub-1"})
    ]


def test_clip_analysis_teardown_retries_past_failures_until_an_attempt_succeeds() -> None:
    terminate = FlakyTerminate(failures=2)

    assert retry_teardown(PROCESS, terminate, attempts=3) is True
    assert terminate.calls == 3


def test_clip_analysis_teardown_stops_after_exactly_the_bounded_attempts() -> None:
    terminate = FlakyTerminate(failures=10)

    assert retry_teardown(PROCESS, terminate, attempts=3) is False
    assert terminate.calls == 3


def test_bed_zone_runner_construction_failure_surfaces_as_the_typed_unavailable_error() -> None:
    recognizer = NvidiaBedZoneRecognizer(RefusingServingClient(), timeout_s=5.0)

    with pytest.raises(BedZoneRecognizerUnavailableError) as raised:
        recognizer(SimpleNamespace(shape=(4, 4, 3)))

    assert isinstance(raised.value.__cause__, ConnectionError)


def test_evidence_export_sender_start_failure_is_a_fatal_delivery_error() -> None:
    runtime = object.__new__(WorkerRuntime)
    runtime._evidence_export_runtime = SimpleNamespace(start_sender=raising(OSError("disk")))

    with pytest.raises(EvidenceDeliveryError) as raised:
        runtime._start_export_sender()

    assert isinstance(raised.value.__cause__, OSError)


def test_evidence_export_sender_refuses_to_start_when_delivery_was_never_composed() -> None:
    runtime = object.__new__(WorkerRuntime)
    runtime._evidence_export_runtime = None

    with pytest.raises(EvidenceDeliveryError, match="not composed"):
        runtime._start_export_sender()


def test_native_heartbeat_relay_failure_for_one_camera_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporters = {"camera-a": Reporter(ConnectionError("relay down")), "camera-b": Reporter()}
    monkeypatch.setattr(
        worker_runtime, "HeartbeatReporter", lambda worker, camera: reporters[camera.camera_id]
    )
    cameras = [SimpleNamespace(camera_id=camera_id) for camera_id in reporters]
    pumps = [CountingPump(camera_id) for camera_id in reporters]
    loop = NativeHeartbeatLoop(object(), cameras, pumps, tick_sec=0.0)
    reporters["camera-b"].after_mark = loop.stop

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert reporters["camera-a"].marked == ["camera-a"]
    assert reporters["camera-b"].marked == ["camera-b"]


def test_native_heartbeat_does_not_report_a_camera_whose_pump_never_advanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = Reporter()
    monkeypatch.setattr(worker_runtime, "HeartbeatReporter", lambda worker, camera: reporter)
    stalled = SimpleNamespace(camera_id="camera-a", processed_count=7)
    camera = SimpleNamespace(camera_id="camera-a")
    loop = NativeHeartbeatLoop(object(), [camera], [stalled], tick_sec=0.01)
    timer = threading.Timer(0.2, loop.stop)

    timer.start()
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()
    loop_thread.join(timeout=5.0)

    assert not loop_thread.is_alive()
    assert reporter.marked == []
