from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.runtime.threads import start_guarded_thread

FRAMES = 5


class _Metadata:
    def __init__(self, pump: NativePolicyPump) -> None:
        self.remaining = FRAMES
        self.pump = pump
        self.drained = threading.Event()

    def subscribe(self, binding: object) -> object:
        return SimpleNamespace(binding=binding, native_publish_sequence=0)

    def wait_accepted(self, token: object, timeout_sec: float) -> object:
        if self.remaining == 0:
            self.drained.set()
            self.pump.stop()
            raise TimeoutError
        self.remaining -= 1
        return SimpleNamespace(native_publish_sequence=FRAMES - self.remaining)

    def expected_binding(self, camera_id: str) -> None:
        return None


def _failing_pump() -> tuple[NativePolicyPump, _Metadata]:
    pump = object.__new__(NativePolicyPump)
    pump._binding = SimpleNamespace(camera_id="camera-a")
    pump._stop = threading.Event()
    pump._observation_coverage = SimpleNamespace(
        observe=lambda frame: None, detect_gap=lambda: None
    )
    pump._diagnostics = SimpleNamespace(record_native_detection_attempt=lambda camera_id: None)
    pump._execution_records = None
    pump._replay_trace = None
    pump._preview_states_lock = threading.Lock()
    pump._preview_states = {}
    pump.processed_count = 0
    pump.failure_count = 0
    pump._failure_logged_at = None
    pump._failures_suppressed = 0

    def boom(frame: object) -> None:
        raise KeyError("missing")

    pump._process = boom
    metadata = _Metadata(pump)
    pump._metadata = metadata
    return pump, metadata


def test_pump_survives_unexpected_exception_and_logs_one_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pump, metadata = _failing_pump()
    deaths: list[str] = []
    with caplog.at_level(logging.WARNING):
        thread = start_guarded_thread("pump", pump.run, lambda: deaths.append("dead"))
        assert metadata.drained.wait(5.0)
        thread.join(5.0)
    assert deaths == []
    assert pump.failure_count == FRAMES
    assert pump.processed_count == FRAMES
    tracebacks = [record for record in caplog.records if record.exc_info]
    assert len(tracebacks) == 1
    assert "camera_id=camera-a" in tracebacks[0].getMessage()


def test_guarded_thread_reports_dying_target(caplog: pytest.LogCaptureFixture) -> None:
    died = threading.Event()

    def target() -> None:
        raise KeyError("boom")

    with caplog.at_level(logging.ERROR):
        start_guarded_thread("dying", target, died.set).join(5.0)
    assert died.is_set()
    assert any("stage=thread:dying" in record.getMessage() for record in caplog.records)


def test_guarded_thread_clean_return_is_not_death() -> None:
    died = threading.Event()
    start_guarded_thread("clean", lambda: None, died.set).join(5.0)
    assert not died.is_set()
