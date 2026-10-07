import logging
import threading
from pathlib import Path

import pytest

from worker.pipeline.output.evidence import evidence_runtime
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_sender import SenderStep


class Idle:
    def run_once(self) -> SenderStep:
        return SenderStep.IDLE


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeWake:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.waits: list[float] = []

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.clock.now += timeout
        return False

    def clear(self) -> None:
        pass

    def set(self) -> None:
        pass


class ScriptedSender:
    def __init__(self, failures: int, stop: threading.Event) -> None:
        self.failures = failures
        self.stop = stop
        self.calls = 0

    def run_once(self) -> SenderStep:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("relay down")
        self.stop.set()
        return SenderStep.IDLE


def run(failures: int) -> tuple[ScriptedSender, FakeWake]:
    clock = FakeClock()
    runtime = EvidenceExportRuntime(store_dir=Path(), queue_directory=Path(), sender=Idle())
    runtime._stop_event = threading.Event()
    wake = FakeWake(clock)
    runtime._wake_sender = wake
    if "monotonic" in EvidenceExportRuntime.__dataclass_fields__:
        runtime.monotonic = clock
    sender = ScriptedSender(failures, runtime._stop_event)
    runtime.sender = sender
    runtime._run_sender()
    return sender, wake


def records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.INFO]


def test_ten_consecutive_failures_log_once_then_recovery_logs_the_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        sender, wake = run(10)
    assert sender.calls == 11
    assert wake.waits == [1.0] * 11
    logged = records(caplog)
    failing = [r for r in logged if r.levelno == logging.WARNING]
    assert 1 <= len(failing) <= 2
    assert "failures=1" in failing[0].getMessage()
    recovered = [r for r in logged if "recovered" in r.getMessage()]
    assert len(recovered) == 1
    assert "failures=10" in recovered[0].getMessage()


def test_a_long_outage_repeats_at_most_once_per_interval_with_the_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    interval = evidence_runtime.SENDER_FAILURE_LOG_INTERVAL_SECONDS
    failures = int(interval * 2) + 5
    with caplog.at_level(logging.INFO):
        run(failures)
    failing = [r for r in records(caplog) if r.levelno == logging.WARNING]
    assert len(failing) == 3
    counts = [r.getMessage().rsplit("failures=", 1)[1].split()[0] for r in failing]
    assert counts == ["1", str(int(interval) + 1), str(int(interval * 2) + 1)]
