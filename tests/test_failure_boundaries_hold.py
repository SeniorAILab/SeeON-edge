from collections.abc import Callable
from types import SimpleNamespace

import pytest

from backend.app.features.evidence import outbox_dispatch
from worker.runtime.clip_analysis_lifecycle import retry_teardown
from worker.runtime.nvidia_bed_zone_recognizer import (
    BedZoneRecognizerUnavailableError,
    NvidiaBedZoneRecognizer,
)
from worker.runtime.worker import EvidenceDeliveryError, NativeHeartbeatLoop, WorkerRuntime

PROCESS = object()


def raising(error: BaseException) -> Callable[..., None]:
    def call(*args: object, **kwargs: object) -> None:
        raise error

    return call


class FlakyTerminate:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, process: object) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("teardown failed")


class StopAfter:
    def __init__(self, ticks: int) -> None:
        self.ticks = ticks

    def wait(self, timeout: float) -> bool:
        self.ticks -= 1
        return self.ticks < 0


class Reporter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.marked: list[str] = []

    def mark_ready(self, camera_id: str) -> None:
        self.marked.append(camera_id)
        if self.error is not None:
            raise self.error


class RefusingServingClient:
    def create(self, *args: object, **kwargs: object) -> object:
        raise ConnectionError("serving endpoint down")


def heartbeat_loop(reporters: dict[str, Reporter]) -> NativeHeartbeatLoop:
    loop = object.__new__(NativeHeartbeatLoop)
    loop._stop = StopAfter(1)
    loop._tick_sec = 0.0
    loop._pumps = [SimpleNamespace(camera_id=camera, processed_count=1) for camera in reporters]
    loop._seen = dict.fromkeys(reporters, 0)
    loop._reporters = reporters
    return loop


def test_outbox_send_that_raises_is_recorded_as_unknown_and_kept_for_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[tuple[object, dict[str, object]]] = []
    client = SimpleNamespace(send_alert_receipt=raising(ConnectionError("reset")))
    monkeypatch.setattr(outbox_dispatch, "scoped_client", lambda raw, camera: client)
    monkeypatch.setattr(outbox_dispatch, "alert_kwargs", lambda envelope: {})
    monkeypatch.setattr(
        outbox_dispatch,
        "_finish",
        lambda delivery, claim, outcome, **fields: finished.append((outcome, fields)),
    )
    claim = SimpleNamespace(backend_camera_id="camera-a", envelope={})

    result = outbox_dispatch.dispatch(object(), claim, object())

    assert result.disposition is outbox_dispatch.DeliveryDisposition.RETRY
    assert result.code == "TRANSPORT_ERROR"
    assert finished == [(outbox_dispatch.DeliveryOutcome.UNKNOWN, {"reason": "TRANSPORT_ERROR"})]


def test_clip_analysis_teardown_retries_past_failures_until_an_attempt_succeeds() -> None:
    terminate = FlakyTerminate(failures=2)

    assert retry_teardown(PROCESS, terminate, attempts=3) is True
    assert terminate.calls == 3


def test_clip_analysis_teardown_stops_after_exactly_the_bounded_attempts() -> None:
    terminate = FlakyTerminate(failures=10)

    assert retry_teardown(PROCESS, terminate, attempts=3) is False
    assert terminate.calls == 3


def test_bed_zone_runner_construction_failure_surfaces_as_the_typed_unavailable_error() -> None:
    recognizer = NvidiaBedZoneRecognizer(RefusingServingClient(), timeout_s=1.0)

    with pytest.raises(BedZoneRecognizerUnavailableError) as raised:
        recognizer._get_runner()

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


def test_native_heartbeat_relay_failure_for_one_camera_does_not_stop_the_others() -> None:
    failing = Reporter(ConnectionError("relay down"))
    healthy = Reporter()

    heartbeat_loop({"camera-a": failing, "camera-b": healthy}).run()

    assert failing.marked == ["camera-a"]
    assert healthy.marked == ["camera-b"]
