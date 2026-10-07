import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI

from backend.app.features.cameras import router as cameras_router
from backend.app.features.connection import router as connection_router
from worker.adapters.device.cuda.probe import probe_cuda_capability
from worker.adapters.device.mps.probe import probe_mps_capability
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_sender import SenderStep
from worker.runtime.faults.record import _frame_hash, make_fault_record


def contained(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "shared.boundary"]


def boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "module", [cameras_router, connection_router], ids=["cameras", "connection"]
)
def test_roster_sync_failure_is_contained_and_logged(
    module: object, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "sync_camera_roster", boom)
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        assert module._trigger_roster_sync(FastAPI()) is None
    assert contained(caplog) == [
        (
            "contained failure boundary=optional_feature stage=roster_sync "
            "exception_class=RuntimeError"
        )
    ]


def fake_cuda(**overrides: object) -> SimpleNamespace:
    cuda = SimpleNamespace(
        get_arch_list=lambda: ("sm_87",),
        device_count=lambda: 1,
        is_available=lambda: False,
    )
    for name, value in overrides.items():
        setattr(cuda, name, value)
    return SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize(
    ("override", "stage", "field", "default"),
    [
        ("get_arch_list", "cuda_arch_list", "arch_list", ()),
        ("device_count", "cuda_device_count", "device_count", 0),
    ],
)
def test_cuda_probe_step_failure_keeps_default_and_logs(
    override: str, stage: str, field: str, default: object, caplog: pytest.LogCaptureFixture
) -> None:
    torch = fake_cuda(**{override: boom})
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        capability = probe_cuda_capability(importer=lambda: torch)
    assert getattr(capability, field) == default
    assert contained(caplog) == [
        f"contained failure boundary=optional_feature stage={stage} exception_class=RuntimeError"
    ]


def test_mps_is_built_failure_keeps_false_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=boom, is_available=lambda: False))
    )
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        capability = probe_mps_capability(importer=lambda: torch)
    assert capability.available is False
    assert contained(caplog) == [
        (
            "contained failure boundary=optional_feature stage=mps_is_built "
            "exception_class=RuntimeError"
        )
    ]


class Idle:
    def run_once(self) -> SenderStep:
        return SenderStep.IDLE


class RaisingOnceSender:
    def __init__(self, runtime_stop: threading.Event) -> None:
        self.stop = runtime_stop
        self.calls = 0

    def run_once(self) -> SenderStep:
        self.calls += 1
        self.stop.set()
        raise RuntimeError("boom")


def test_evidence_sender_tick_failure_is_logged_and_loop_keeps_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = EvidenceExportRuntime(store_dir=Path(), queue_directory=Path(), sender=Idle())
    runtime._stop_event = threading.Event()
    runtime._wake_sender = threading.Event()
    runtime._wake_sender.set()
    sender = RaisingOnceSender(runtime._stop_event)
    runtime.sender = sender
    with caplog.at_level(logging.WARNING):
        runtime._run_sender()
    assert sender.calls == 1
    assert not runtime._wake_sender.is_set()
    assert [r.getMessage() for r in caplog.records] == [
        "evidence sender tick failing exception_class=RuntimeError failures=1"
    ]


class BrokenFrame(np.ndarray):
    def tobytes(self, order: str = "C") -> bytes:
        raise RuntimeError("boom")

    @property
    def shape(self) -> tuple[int, ...]:
        raise RuntimeError("boom")


def broken_frame() -> np.ndarray:
    return np.zeros((2, 2), dtype=np.uint8).view(BrokenFrame)


def test_frame_hash_failure_returns_none_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        assert _frame_hash(broken_frame()) is None
    assert contained(caplog) == [
        (
            "contained failure boundary=optional_feature stage=fault_frame_hash "
            "exception_class=RuntimeError"
        )
    ]


def test_fault_record_shape_failure_keeps_none_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        record = make_fault_record(
            RuntimeError("x"), profile="p", task="t", stage="s", camera_id="c", image=broken_frame()
        )
    assert record.frame_shape is None
    assert (
        "contained failure boundary=optional_feature stage=fault_frame_shape "
        "exception_class=RuntimeError"
    ) in contained(caplog)
