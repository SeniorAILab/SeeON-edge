"""Focused contracts for local metrics and the frozen relay projection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from contracts.decode_diagnostics import DecodeSelection
from worker.runtime.telemetry.runtime_diagnostics import (
    EncoderLifecycleSnapshot,
    WorkerDiagnostics,
)
from worker.runtime.telemetry.status_store import CameraStatus, StatusStore


@dataclass(frozen=True, slots=True)
class _BusMetrics:
    published: int
    taken: int
    dropped: int
    queue_age_sec: float


@dataclass(frozen=True, slots=True)
class _Bus:
    values: dict[str, _BusMetrics]

    def metrics(self, name: str) -> _BusMetrics:
        return self.values[name]


def _selection() -> DecodeSelection:
    return DecodeSelection(
        requested="opencv",
        selected="opencv",
        fallback_count=0,
        last_reason=None,
        updated_at_sec=10.0,
    )


def test_local_snapshot_includes_stage_bus_encoder_and_failure_metrics() -> None:
    # Given
    statuses = StatusStore()
    _ = statuses.set_status(
        "camera-1",
        "facility-1",
        CameraStatus.DEGRADED,
        error_category="auth",
        timestamp=10.0,
    )
    diagnostics = WorkerDiagnostics(statuses)
    diagnostics.register_decode("camera-1", "opencv")
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "decode", 0.1)
    diagnostics.record_stage_timing("camera-1", "decode", 0.2)
    diagnostics.register_bus(
        "camera-1",
        _Bus({"inference": _BusMetrics(5, 3, 2, 0.4)}),
        ("inference",),
    )
    diagnostics.update_encoder_lifecycle(
        EncoderLifecycleSnapshot(
            process_starts=4,
            recreates=2,
            failures=1,
            active_sessions=3,
            finalized_segments=8,
            unavailable_cameras=("camera-1",),
        )
    )

    # When
    snapshot = diagnostics.snapshot()

    # Then
    camera = snapshot.cameras[0]
    assert camera.failure_category == "auth"
    assert camera.stage_timings[0].samples == 2
    assert abs(camera.stage_timings[0].total_sec - 0.3) < 1e-12
    assert camera.stage_timings[0].last_sec == 0.2
    assert camera.bus[0].dropped == 2
    assert camera.bus[0].queue_age_sec == 0.4
    assert snapshot.encoder.failures == 1
    assert snapshot.encoder.finalized_segments == 8
    assert snapshot.encoder.unavailable_cameras == ("camera-1",)


def test_local_metric_is_excluded_from_frozen_wire_payload() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "local_metric_must_not_leak", 1.0)

    # When
    payload = diagnostics.to_payload("facility-1", None, 1)

    # Then
    assert set(payload) == {"facility_id", "generation", "seq", "cameras", "clip_recorder"}
    assert set(payload["cameras"][0]) == {"camera_id", "decode"}
    assert "local_metric_must_not_leak" not in repr(payload)


def test_structured_log_contains_local_metrics(caplog: pytest.LogCaptureFixture) -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "inference", 0.25)

    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()

    # Then
    record = caplog.records[-1]
    assert record.getMessage() == "worker.runtime.telemetry"
    assert vars(record).get("camera_id") == "camera-1"
    assert vars(record).get("stage_timings") == {
        "inference": {
            "samples": 1,
            "total_sec": 0.25,
            "last_sec": 0.25,
            "max_sec": 0.25,
        }
    }
