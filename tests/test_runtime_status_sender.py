from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import final

import pytest

from contracts.decode_diagnostics import DecodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.inference_telemetry import (
    CameraInferenceTelemetry,
    InferenceTelemetrySnapshot,
)
from worker.pipeline.perception.scene_state import BedRegionCacheCounters
from worker.runtime.flow.lifecycle_supervisor import FlowLifecycleSupervisor
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RuntimeStatusSender,
    RuntimeStatusSenderConfig,
)
from worker.runtime.telemetry.wire import (
    ClipRecorderStatus,
    RelayRuntimeStatusPayload,
)


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


class _StaticInferenceSource:
    def __init__(self, cameras: dict[str, CameraInferenceTelemetry]) -> None:
        self._cameras = cameras

    def snapshot(self) -> InferenceTelemetrySnapshot:
        return InferenceTelemetrySnapshot(
            cameras=self._cameras,
            batch_sizes={},
            forward_p50_sec=0.0,
            forward_p95_sec=0.0,
        )


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
            "detection": {
                "expected": False,
                "inference_admitted": 0,
                "inference_succeeded": 0,
                "inference_overwritten": 0,
                "decision_completed": 0,
            },
        }
    ]
    assert sender.generation == 7
    assert sender.is_alive is False


def test_sender_payload_projects_inference_and_decision_counters() -> None:
    diagnostics = _diagnostics()
    diagnostics.register_inference(
        _StaticInferenceSource(
            {
                "camera-a": CameraInferenceTelemetry(
                    admitted=5,
                    overwritten=1,
                    inferred=4,
                    queue_age_sec=0.2,
                )
            }
        )
    )
    diagnostics.record_detection_completed("camera-a")
    diagnostics.record_detection_completed("camera-a")
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(diagnostics, "facility-a", transport)

    assert sender.publish_once() is True
    detection = transport.payloads[0]["cameras"][0].get("detection")
    assert detection == {
        "expected": True,
        "inference_admitted": 5,
        "inference_succeeded": 4,
        "inference_overwritten": 1,
        "decision_completed": 2,
    }


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


def test_before_publish_hook_refreshes_diagnostics_on_every_tick() -> None:
    diagnostics = _diagnostics()
    calls = {"count": 0}

    def before_publish() -> None:
        calls["count"] += 1
        diagnostics.set_clip_recorder_status(
            ClipRecorderStatus(available=True, finalized_clips=calls["count"])
        )

    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        diagnostics,
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
        before_publish=before_publish,
    )

    sender.start()
    try:
        _wait_until(lambda: len(transport.payloads) >= 2)
    finally:
        sender.stop()

    assert calls["count"] >= 2
    assert transport.payloads[-1]["clip_recorder"]["finalized_clips"] == calls["count"]


def test_sender_logs_a_local_diagnostics_snapshot_on_its_own_tick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    diagnostics = _diagnostics()
    counters = BedRegionCacheCounters(fresh=2)
    diagnostics.record_bed_region("camera-a", BedRegionCacheState.FRESH, counters.snapshot())
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        diagnostics,
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    with caplog.at_level(logging.INFO):
        sender.start()
        try:
            _wait_until(lambda: bool(transport.payloads))
        finally:
            sender.stop()

    telemetry_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("worker.runtime.telemetry ")
    ]
    assert telemetry_records
    assert vars(telemetry_records[-1]).get("camera_id") == "camera-a"
    assert vars(telemetry_records[-1]).get("bed_region", {}).get("freshness") == "fresh"


@final
class _LogSnapshotAlwaysFailsDiagnostics:
    __slots__ = ("_inner",)

    def __init__(self, inner: WorkerDiagnostics) -> None:
        self._inner = inner

    def to_payload(
        self, facility_id: str, generation: int | None, seq: int
    ) -> RelayRuntimeStatusPayload:
        return self._inner.to_payload(facility_id, generation, seq)

    def to_payloads(
        self, camera_facilities: object, generation: int | None, seq: int
    ) -> list[RelayRuntimeStatusPayload]:
        return self._inner.to_payloads(camera_facilities, generation, seq)  # type: ignore[arg-type]

    def log_snapshot(self) -> None:
        raise RuntimeError("boom")


def test_sender_survives_a_log_snapshot_failure_and_keeps_delivering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        _LogSnapshotAlwaysFailsDiagnostics(_diagnostics()),  # type: ignore[arg-type]
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    with caplog.at_level(logging.WARNING):
        sender.start()
        try:
            _wait_until(lambda: bool(transport.payloads))
        finally:
            sender.stop()

    assert transport.payloads
    assert any("log_snapshot" in record.getMessage() for record in caplog.records)


@dataclass(frozen=True, slots=True)
class _PlaneStatus:
    fatal_error: str | None


@final
class _ScriptedPlane:
    __slots__ = ("_steps",)

    def __init__(self, *steps: Exception | str | None) -> None:
        self._steps = list(steps)

    def status(self) -> _PlaneStatus:
        step = self._steps.pop(0) if len(self._steps) > 1 else self._steps[0]
        if isinstance(step, Exception):
            raise step
        return _PlaneStatus(fatal_error=step)

    def published_frames(self, _camera_id: str) -> int:
        return 0

    def source_failure(self, _camera_id: str, _category: str) -> object:
        return None

    def clear_preview(self, _camera_id: str) -> None:
        return None


@pytest.mark.parametrize(
    "healthy_ticks_before_failure",
    [pytest.param((), id="startup-publish"), pytest.param((None,), id="loop-tick")],
)
def test_a_failing_lifecycle_tick_is_logged_and_the_next_tick_still_reports_fatal(
    healthy_ticks_before_failure: tuple[None, ...],
    caplog: pytest.LogCaptureFixture,
) -> None:
    tick_error = RuntimeError("media plane status unavailable")
    fatal_errors: list[str] = []
    lifecycle = FlowLifecycleSupervisor(
        _ScriptedPlane(*healthy_ticks_before_failure, tick_error, "xid 79"),
        ["camera-a"],
        on_ready=lambda _camera_id: None,
        on_unready=lambda _camera_id: None,
        on_fatal=fatal_errors.append,
    )
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        _diagnostics(),
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
        before_publish=lifecycle.tick,
    )

    with caplog.at_level(logging.WARNING):
        sender.start()
        try:
            _wait_until(lambda: bool(fatal_errors) and bool(transport.payloads))
        finally:
            sender.stop()

    assert fatal_errors == ["xid 79"]
    logged_failures = [record for record in caplog.records if record.exc_info is not None]
    assert [(record.levelno, record.exc_info[1]) for record in logged_failures] == [
        (logging.ERROR, tick_error)
    ]


def _wait_until(predicate, timeout_sec: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for runtime status sender")
