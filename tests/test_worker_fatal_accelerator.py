from __future__ import annotations

import threading
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np
import pytest

from shared.events.delivery_queue import DeliveryQueue
from worker.adapters.model.errors import FatalAcceleratorError, ModelInputError
from worker.adapters.model.yolo_api import (
    YoloPredictOptions,
    _classify_or_reraise,
    predict_one,
)
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE, FaultHandler
from worker.runtime.faults.record import (
    FirstFaultRecord,
    make_fault_record,
    persist_first_fault,
)


def _record(**kwargs) -> FirstFaultRecord:
    defaults = {
        "pid": 1,
        "boot_time_iso": "2026-01-01T00:00:00Z",
        "profile": "cuda",
        "task": "pose",
        "stage": "inference",
        "camera_id": "cam-1",
        "frame_index": 42,
        "pts": 1.0,
        "frame_shape": (480, 640, 3),
        "frame_hash_sha256": None,
        "model_artifact_digest": "abc123",
        "invocation_seq": 1,
        "exception_type": "RuntimeError",
        "exception_message": "CUDA error: device-side assert triggered",
        "exit_code": 4,
        "fault_time_iso": "2026-01-01T00:00:01Z",
    }
    defaults.update(kwargs)
    return FirstFaultRecord(**defaults)


def _queued_fault(tmp_path: Path) -> dict[str, object]:
    queue = DeliveryQueue(tmp_path / "delivery-queue")
    deadline = time_module.monotonic() + 1.0
    while time_module.monotonic() < deadline:
        entries = tuple(queue.entries())
        if entries:
            return entries[0]
    pytest.fail("first-fault queue entry was not durably admitted")


@final
@dataclass
class _FakeLoop:
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True


def test_cuda_keyword_raises_fatal_accelerator_error() -> None:
    exc = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(FatalAcceleratorError) as exc_info:
        _classify_or_reraise(exc, task="pose", camera_id="cam-1")
    assert exc_info.value.task == "pose"
    assert exc_info.value.camera_id == "cam-1"


def test_non_cuda_error_raises_yolo_forward_error() -> None:
    from worker.adapters.model.yolo_api import YoloForwardError

    exc = OSError("model file not found")
    with pytest.raises(YoloForwardError):
        _classify_or_reraise(exc, task="pose", camera_id="cam-1")


def test_cuda_classification_is_case_insensitive() -> None:
    for msg in ("cuda launch failed", "CUDA error", "cuLaunch error", "device lost"):
        exc = RuntimeError(msg)
        with pytest.raises(FatalAcceleratorError):
            _classify_or_reraise(exc, task="pose", camera_id="cam-1")


def test_non_fatal_validation_error_stays_isolated() -> None:
    err = ModelInputError("wrong frame shape")
    assert not isinstance(err, FatalAcceleratorError)


def test_predict_one_raises_fatal_on_cuda_error() -> None:
    options = YoloPredictOptions(task="pose", confidence=0.05, device="cuda")

    class _BadModel:
        def predict(self, **_kwargs):
            raise RuntimeError("CUDA error: illegal memory access")

    with pytest.raises(FatalAcceleratorError):
        predict_one(_BadModel(), np.zeros((4, 4, 3), dtype=np.uint8), options, camera_id="cam-1")


def test_predict_one_raises_forward_error_on_non_cuda() -> None:
    from worker.adapters.model.yolo_api import YoloForwardError

    options = YoloPredictOptions(task="bed", confidence=0.25, device="cpu")

    class _BadModel:
        def predict(self, **_kwargs):
            raise OSError("onnx model corrupt")

    with pytest.raises(YoloForwardError):
        predict_one(_BadModel(), np.zeros((4, 4, 3), dtype=np.uint8), options)


def test_persist_first_fault_admits_exactly_one_queue_record(tmp_path: Path) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    rec = _record()
    wrote_first = persist_first_fault(rec, state_dir=tmp_path)
    wrote_second = persist_first_fault(rec, state_dir=tmp_path)

    assert wrote_first is True
    assert wrote_second is False

    queue = DeliveryQueue(tmp_path / "delivery-queue")
    entries = tuple(queue.entries())
    assert len(entries) == 1
    assert entries[0]["event_type"] == "runtime.fault"


def test_persist_first_fault_degrades_to_false_when_queue_parent_is_uncreatable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")
    state_dir = blocker / "state"

    rec = _record()
    with caplog.at_level("WARNING"):
        written = persist_first_fault(rec, state_dir=state_dir)

    assert written is False


def test_persist_first_fault_is_independent_of_the_delivery_queue(tmp_path: Path) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    rec = _record()
    started = time_module.monotonic()
    written = persist_first_fault(rec, state_dir=tmp_path)
    elapsed = time_module.monotonic() - started

    assert written is True
    assert elapsed < 1.0


def test_persist_first_fault_writes_to_production_delivery_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    monkeypatch.setattr(mod, "resolve_state_dir", lambda: tmp_path)

    rec = _record(camera_id="cam-edge-1", exception_message="CUDA error: edge path")
    wrote_first = persist_first_fault(rec)
    wrote_second = persist_first_fault(rec)

    assert wrote_first is True
    assert wrote_second is False

    assert _queued_fault(tmp_path)["camera_id"] == "cam-edge-1"


def test_persist_first_fault_returns_immediately_under_held_queue_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    monkeypatch.setattr(mod, "resolve_state_dir", lambda: tmp_path)

    queue = DeliveryQueue(tmp_path / "delivery-queue")
    holder = queue._lock_path.open("a+b")
    import fcntl

    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

    try:
        rec = _record()
        started = time_module.monotonic()
        with caplog.at_level("WARNING"):
            written = persist_first_fault(rec)
        elapsed = time_module.monotonic() - started
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    assert written is False
    assert elapsed < 1.0
    assert "queue admission lock_unavailable" in caplog.text


def test_persist_first_fault_includes_frame_hash(tmp_path: Path) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    rec = make_fault_record(
        RuntimeError("CUDA error"),
        profile="cuda",
        task="pose",
        stage="inference",
        camera_id="cam-1",
        image=image,
    )
    assert rec.frame_hash_sha256 is not None
    assert len(rec.frame_hash_sha256) == 64


def test_fault_handler_stops_all_loops_and_exits(tmp_path: Path) -> None:
    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=tmp_path)
    loop_a = _FakeLoop()
    loop_b = _FakeLoop()
    handler.register_loop(loop_a)
    handler.register_loop(loop_b)

    rec = _record()
    import worker.runtime.faults.record as mod

    mod._written = False

    handler.handle(FatalAcceleratorError("CUDA error"), rec)

    assert loop_a.stopped
    assert loop_b.stopped
    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fault_handler_is_idempotent(tmp_path: Path) -> None:
    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=tmp_path)
    loop = _FakeLoop()
    handler.register_loop(loop)

    import worker.runtime.faults.record as mod

    mod._written = False

    rec = _record()
    t1 = threading.Thread(target=handler.handle, args=(FatalAcceleratorError("CUDA error"), rec))
    t2 = threading.Thread(target=handler.handle, args=(FatalAcceleratorError("CUDA error"), rec))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fault_handler_exits_even_when_fault_storage_is_unavailable(tmp_path: Path) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False

    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")
    state_dir = blocker / "state"

    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=state_dir)
    loop = _FakeLoop()
    handler.register_loop(loop)

    rec = _record()
    handler.handle(FatalAcceleratorError("CUDA error"), rec)

    assert loop.stopped
    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fatal_exit_code_is_4() -> None:
    assert FATAL_ACCELERATOR_EXIT_CODE == 4


def test_fatal_accelerator_propagates_through_ingest_loop() -> None:
    class _FatalBus:
        def publish(self, _packet) -> None:
            raise FatalAcceleratorError("CUDA error: illegal memory access")

    class _SimpleLoop:
        camera_id = "cam-1"
        _ready = False

        def run(self) -> None:
            bus = _FatalBus()
            for _ in range(1):
                if not self._ready:
                    self._ready = True
                try:
                    bus.publish(None)
                except FatalAcceleratorError:
                    raise
                except Exception:  # noqa: BLE001, S110
                    pass

    loop = _SimpleLoop()
    with pytest.raises(FatalAcceleratorError):
        loop.run()
