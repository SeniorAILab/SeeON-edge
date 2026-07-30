from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from edge_worker_fixtures import edge_config_payload

from contracts.frame import Frame
from contracts.runner import pose_result
from edge.runtime import edge_worker
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker_config import EdgeWorkerConfig
from edge.runtime.scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 3
    stride: int = 1


@dataclass(frozen=True, slots=True)
class _FallModel:
    metadata: _FallMetadata = field(default_factory=_FallMetadata)
    operating_threshold: float = 0.5

    def predict(self, features: np.ndarray) -> float:
        del features
        return 0.91


@dataclass(slots=True)
class _PoseRunner:
    def run(self, image: np.ndarray):
        del image
        keypoints = tuple((20, 20, 0.9) for _ in range(17))
        return pose_result((keypoints,), ((10, 10, 40, 60, 0.95),))


@dataclass(slots=True)
class _Sink:
    events: list[dict[str, object]] = field(default_factory=list)

    def emit(self, event: dict[str, object]) -> None:
        self.events.append(event)

@dataclass(slots=True)
class _OverlaySink:
    observations: list[object] = field(default_factory=list)

    def publish(
        self, camera_id: str, frame: Frame, observation: object, debug_snapshots: object
    ) -> None:
        del camera_id, frame, debug_snapshots
        self.observations.append(observation)



def test_worker_preserves_fall_labels_across_all_enabled_domain_detectors() -> None:
    sink = _Sink()
    overlay_sink = _OverlaySink()
    config_payload = edge_config_payload(camera_count=1)
    config_payload["domains"] = {"enabled": ["fall", "bed_exit"]}  # type: ignore[typeddict-unknown-key]
    config = EdgeWorkerConfig.model_validate(config_payload)
    detectors = edge_worker._domain_detectors(config, _FallModel())
    worker = CameraWorker(
        camera_id="camera-1",
        facility_id="facility-1",
        frame_source=(),
        runners={"pose": _PoseRunner()},
        scheduler=Scheduler({"pose": 1}),
        domain_detectors=detectors,
        overlay_sink=overlay_sink,
        event_sink=sink,
    )

    returned = tuple(
        worker.process_frame(
            Frame(index=index, time_sec=float(index), image=np.zeros((80, 80, 3), dtype=np.uint8))
        )
        for index in range(3)
    )
    assert len(sink.events) == 1
    assert returned[-1].labels[0].is_fall
    assert overlay_sink.observations[-1].labels[0].is_fall
    assert worker.scene_state.latest_observation is not None
    assert worker.scene_state.latest_observation.labels[0].is_fall
    event = dict(sink.events[0])
    assert UUID(str(event.pop("edge_event_id"))).version == 4
    assert str(event.pop("detected_at")).endswith("Z")
    assert event == {
            "domain": "fall",
            "event_type": "fall",
            "identity": 1,
            "probability": 0.91,
            "time_sec": 2.0,
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "clip_id": None,
            "audit": {
                "clock_source": "edge_wall_clock",
                "operating_threshold": 0.5,
            },
    }
    reversed_sink = _Sink()
    reversed_overlay_sink = _OverlaySink()
    reversed_worker = CameraWorker(
        camera_id="camera-2",
        facility_id="facility-1",
        frame_source=(),
        runners={"pose": _PoseRunner()},
        scheduler=Scheduler({"pose": 1}),
        domain_detectors=tuple(reversed(edge_worker._domain_detectors(config, _FallModel()))),
        overlay_sink=reversed_overlay_sink,
        event_sink=reversed_sink,
    )
    reversed_returned = tuple(
        reversed_worker.process_frame(
            Frame(index=index, time_sec=float(index), image=np.zeros((80, 80, 3), dtype=np.uint8))
        )
        for index in range(3)
    )
    assert reversed_returned[-1].labels[0].is_fall
    assert reversed_overlay_sink.observations[-1].labels[0].is_fall
    assert reversed_worker.scene_state.latest_observation is not None
    assert reversed_worker.scene_state.latest_observation.labels[0].is_fall