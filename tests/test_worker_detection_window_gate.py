from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import pytest

import worker.runtime.worker as worker_module
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallDomainDecider, FallPolicyDecider, FallProbabilities
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.worker import CameraDetectionPlan, WorkerRuntime
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


@final
class _RecordingDecider:
    def __init__(self, events: tuple[BusinessEvent, ...] = ()) -> None:
        self.calls = 0
        self._events = events

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        self.calls += 1
        return self._events


@final
class _TraceRecordingDecider:
    def __init__(self, snapshots: tuple[DecisionTraceSnapshot, ...]) -> None:
        self.calls = 0
        self.last_trace_snapshots = snapshots

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        self.calls += 1
        return ()


def _input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=0.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_window_gated_decider_skips_update_and_wrapped_state_outside_window() -> None:
    inner = _RecordingDecider()
    window = DetectionWindow(start="21:00", end="06:00", tz="UTC")
    gated = worker_module._WindowGatedDecider(
        inner, window, clock=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    )

    assert gated.update(_input()) == ()
    assert inner.calls == 0
    assert len(gated.last_trace_snapshots) == 1
    assert gated.last_trace_snapshots[0].current_state == "not-evaluated"
    assert gated.last_trace_snapshots[0].reason == "outside-detection-window"


def test_window_gated_decider_passes_through_inside_window() -> None:
    expected = (
        BusinessEvent(
            domain="fall",
            event_type="fall",
            identity="0",
            camera_id="camera-1",
            facility_id="facility-1",
            time_sec=0.0,
            probability=1.0,
            person_id=0,
        ),
    )
    inner = _RecordingDecider(expected)
    window = DetectionWindow(start="21:00", end="06:00", tz="UTC")
    gated = worker_module._WindowGatedDecider(
        inner, window, clock=lambda: datetime(2026, 1, 1, 23, 0, tzinfo=UTC)
    )

    assert gated.update(_input()) == expected
    assert inner.calls == 1
    assert gated.last_trace_snapshots == ()


def test_window_gated_decider_forwards_trace_snapshots_inside_window() -> None:
    snapshot = DecisionTraceSnapshot(
        reason="below-threshold",
        previous_state="clear",
        current_state="clear",
        triggered=False,
        track_id=4,
        bed_id=None,
        values={"fall_transition_probability": 0.12},
    )
    inner = _TraceRecordingDecider((snapshot,))
    gated = worker_module._WindowGatedDecider(
        inner,
        DetectionWindow(start="21:00", end="06:00", tz="UTC"),
        clock=lambda: datetime(2026, 1, 1, 23, 0, tzinfo=UTC),
    )

    assert gated.update(_input()) == ()
    assert inner.calls == 1
    assert gated.last_trace_snapshots == (snapshot,)


@final
class _UnusedServingClient:
    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"serving client should not be used to build a plan ({task})")


@final
class _FakeFallModel:
    operating_threshold = 0.5

    def predict(self, _features: object) -> FallProbabilities:
        return FallProbabilities(0.0, 0.99, 0.1)


@final
@dataclass(frozen=True, slots=True)
class _LoadedBundle:
    published_weights_digest: str = "a" * 64
    preprocessing_identity: str = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"


def _config(artifact_dir: Path, detection_windows: dict[str, object] | None = None) -> WorkerConfig:
    domains: dict[str, object] = {"enabled": ["fall", "bed_exit"]}
    if detection_windows is not None:
        domains["detection_windows"] = detection_windows
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": domains,
            "models": {
                "fall": {
                    "type": "pose-bbox56-proxy-v0",
                    "framework": "onnxruntime",
                    "mode": "sequence",
                    "artifact_dir": str(artifact_dir),
                    "weights": "model.pt",
                    "architecture": "arch.json",
                    "metadata": "metadata.yaml",
                    "window": 30,
                    "stride": 5,
                    "input_shape": [30, 56],
                    "operating_threshold": 0.5,
                    "schema_version": 2,
                    "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
                }
            },
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                }
            ],
        }
    )


def _plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detection_windows: dict[str, object] | None = None,
) -> CameraDetectionPlan:
    artifact_dir = tmp_path / "fall-bundle"
    artifact_dir.mkdir()
    for name in ("model.pt", "model.onnx", "arch.json", "metadata.yaml"):
        (artifact_dir / name).write_bytes(b"")
    config = _config(artifact_dir, detection_windows)
    runtime = WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=_UnusedServingClient(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        state_dir=tmp_path,
    )
    monkeypatch.setattr(runtime, "_create_fall_model", _FakeFallModel)
    runtime._loaded_fall_bundle = _LoadedBundle()
    profile = PROFILE_REGISTRY["flow"]
    runtime._initialize_flow_policy_graph(
        BootContext(
            profile=profile,
            device=profile.device,
            decode=profile.decode,
            encode=profile.encode,
            requested_profile="flow",
        )
    )
    return runtime._preflight_camera_graph(config.cameras[0])


def test_fall_domain_is_ungated_24_7_when_no_window_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, monkeypatch)

    decider = plan.domain_deciders["fall"]
    assert plan.detection_windows["fall"] is None
    assert isinstance(decider, FallDomainDecider)
    assert isinstance(decider.policy, FallPolicyDecider)


def test_fall_domain_is_gated_by_the_common_wrapper_once_a_window_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, monkeypatch, {"fall": {"start": "21:00", "end": "06:00", "tz": "UTC"}})

    decider = plan.domain_deciders["fall"]
    assert isinstance(decider, worker_module._WindowGatedDecider)
    assert decider.window == DetectionWindow(start="21:00", end="06:00", tz="UTC")
    assert isinstance(decider.decider, FallDomainDecider)
    assert isinstance(decider.decider.policy, FallPolicyDecider)


def test_bed_exit_is_never_wrapped_by_the_common_gate_even_with_a_window_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(
        tmp_path, monkeypatch, {"bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"}}
    )

    decider = plan.domain_deciders["bed_exit"]
    assert plan.detection_windows["bed_exit"] == DetectionWindow(
        start="21:00", end="06:00", tz="UTC"
    )
    assert isinstance(decider, BedExitMonitor)
    assert not isinstance(decider, worker_module._WindowGatedDecider)
