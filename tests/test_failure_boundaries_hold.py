import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Self

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


class StatusSender:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.started = 0

    def __call__(self, *args: object, **kwargs: object) -> Self:
        return self

    def start(self) -> None:
        self.started += 1
        if self.error is not None:
            raise self.error


class ClipStoreLockProbe:
    def __init__(self) -> None:
        self.events: list[str] = []

    def acquire(self, store: Path) -> Self:
        self.events.append("locked")
        return self

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        self.events.append("unlocked")


def relay_config(*camera_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        cameras=tuple(
            SimpleNamespace(camera_id=camera_id, facility_id="facility-a")
            for camera_id in camera_ids
        ),
        relay=SimpleNamespace(
            url="http://relay.test",
            token=SimpleNamespace(get_secret_value=lambda: "relay-token"),
        ),
        version="config-v1",
    )


def status_runtime(tmp_path: Path) -> WorkerRuntime:
    runtime = object.__new__(WorkerRuntime)
    runtime.config = relay_config("camera-a")
    runtime.diagnostics = object()
    runtime._state_dir = tmp_path / "state"
    runtime._runtime_status_sender = None
    return runtime


def test_runtime_status_sender_start_failure_leaves_the_worker_running_without_a_sender(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = StatusSender(ConnectionError("relay down"))
    monkeypatch.setattr(worker_runtime, "RuntimeStatusSender", sender)
    runtime = status_runtime(tmp_path)

    with caplog.at_level(logging.WARNING):
        runtime._start_runtime_status_sender()

    assert sender.started == 1
    assert runtime._runtime_status_sender is None
    [record] = [r for r in caplog.records if "runtime status sender" in r.getMessage()]
    assert record.getMessage() == "runtime status sender failed to start"
    assert record.exc_info is not None


def test_runtime_status_sender_that_starts_is_kept_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sender = StatusSender()
    monkeypatch.setattr(worker_runtime, "RuntimeStatusSender", sender)
    runtime = status_runtime(tmp_path)

    runtime._start_runtime_status_sender()

    assert sender.started == 1
    assert runtime._runtime_status_sender is sender


def test_evidence_delivery_init_failure_under_the_lock_is_a_typed_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = ClipStoreLockProbe()
    export = SimpleNamespace(initialize_under_lock=raising(OSError("queue directory unwritable")))
    monkeypatch.setattr(worker_runtime.EvidenceExportRuntime, "from_config", answering(export))
    monkeypatch.setattr(worker_runtime.ClipStoreLock, "acquire", lock.acquire)
    runtime = object.__new__(WorkerRuntime)
    runtime.config = relay_config("camera-a")
    runtime._execution_record_lanes = None
    runtime._state_dir = tmp_path / "state"
    runtime._clip_export_policy = SimpleNamespace(enabled=True)
    runtime._resolved_clip_store_dir = lambda: tmp_path / "clips"
    runtime._evidence_export_runtime = None

    with pytest.raises(EvidenceDeliveryError, match="under the clip-store lock") as raised:
        runtime._compose_evidence_export()

    assert isinstance(raised.value.__cause__, OSError)
    assert lock.events == ["locked", "unlocked"]
    assert runtime._evidence_export_runtime is None
    with pytest.raises(EvidenceDeliveryError, match="not composed"):
        runtime._start_export_sender()


def provenance_runtime(tmp_path: Path, admission: object) -> WorkerRuntime:
    runtime = object.__new__(WorkerRuntime)
    runtime.config = relay_config()
    runtime._shared_graph = SimpleNamespace(identities=())
    runtime._module_registry = object()
    runtime._module_versions = {}
    runtime._restart_generation = 0
    runtime._build_revision = "build-1"
    runtime._environment_facts_factory = raising(OSError("environment probe failed"))
    runtime._state_dir = tmp_path / "state"
    runtime._boot_instance_id = "boot-1"
    runtime._runtime_manifest = SimpleNamespace(canonical_json="stale")
    runtime._selected_bundle_admission = admission
    return runtime


def test_runtime_provenance_failure_drops_the_manifest_and_lets_activation_continue(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = provenance_runtime(tmp_path, admission=None)

    with caplog.at_level(logging.WARNING):
        runtime._apply_runtime_manifest(SimpleNamespace(), {})

    assert runtime._runtime_manifest is None
    assert not (tmp_path / "state" / "runtime-manifest").exists()
    [record] = [r for r in caplog.records if "runtime provenance" in r.getMessage()]
    assert record.getMessage() == "runtime provenance could not be applied; continuing without it"
    assert record.exc_info is not None


def test_runtime_provenance_failure_is_fatal_once_a_selected_bundle_was_admitted(
    tmp_path: Path,
) -> None:
    runtime = provenance_runtime(tmp_path, admission=object())

    with pytest.raises(OSError, match="environment probe failed"):
        runtime._apply_runtime_manifest(SimpleNamespace(), {})

    assert runtime._runtime_manifest is None
