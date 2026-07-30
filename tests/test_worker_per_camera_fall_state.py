from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts.frame import Frame
from contracts.runner import pose_result
from contracts.tracker import TrackerProtocol
from edge.perception.tracker import GreedyIouTracker
from edge.runtime import edge_worker
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker_config import CameraRuntimeConfig, EdgeWorkerConfig
from edge.runtime.scheduler import Scheduler
from edge.runtime.status_store import StatusStore


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 3
    stride: int = 1
    mode: str = "sequence"


@dataclass(frozen=True, slots=True)
class _FallModel:
    metadata: _FallMetadata = field(default_factory=_FallMetadata)
    operating_threshold: float = 0.5

    def predict(self, features: np.ndarray) -> float:
        return 0.91 if features.shape == (3, 51) else 0.0


class _Runner:
    def run(self, image: np.ndarray):
        del image
        return pose_result((), ())



@dataclass(slots=True)
class _Registry:
    created: dict[str, int] = field(
        default_factory=lambda: {"pose": 0, "person": 0, "bed": 0, "fall": 0}
    )
    fall_model: _FallModel = field(default_factory=_FallModel)
    pose_runner: _Runner = field(default_factory=_Runner)
    person_runner: _Runner = field(default_factory=_Runner)

    bed_runner: _Runner = field(default_factory=_Runner)

    def create(self, task: str, **kwargs: object) -> object:
        del kwargs
        self.created[task] += 1
        if task == "pose":
            return self.pose_runner
        if task == "person":
            return self.person_runner
        if task == "bed":
            return self.bed_runner
        if task == "fall":
            return self.fall_model
        raise AssertionError(task)


class _Stager:
    def stage(self, event) -> None:
        del event

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        del edge_event_id, clip_id


def test_fall_classifier_state_is_per_camera() -> None:
    registry = _Registry()
    supervisor = edge_worker._build_supervisor(
        _two_camera_config(),
        StatusStore(),
        device="cpu",
        decode="opencv",
        registry=registry,
    )

    workers = [loop.worker for loop in supervisor.loops]
    detectors = [worker.domain_detectors[0] for worker in workers]

    assert registry.created == {"pose": 1, "person": 0, "bed": 1, "fall": 1}
    assert isinstance(GreedyIouTracker(), TrackerProtocol)
    assert len({id(worker.tracker) for worker in workers}) == 2
    assert all(not hasattr(detector, "_tracker") for detector in detectors)
    assert len({id(detector) for detector in detectors}) == 2
    classifiers = [detector._prepare_input.__self__ for detector in detectors]
    assert len({id(classifier) for classifier in classifiers}) == 2
    assert all(classifier.model is registry.fall_model for classifier in classifiers)


def test_supervisor_wires_the_camera_specific_durable_stager() -> None:
    registry = _Registry()
    stager = _Stager()

    supervisor = edge_worker._build_supervisor(
        _two_camera_config(),
        StatusStore(),
        device="cpu",
        decode="opencv",
        registry=registry,
        evidence_stagers={"camera-1": stager},
    )

    assert supervisor.loops[0].worker.evidence_stager is stager
    assert supervisor.loops[1].worker.evidence_stager is None

def test_camera_worker_stamps_one_shared_tracker_id_for_fall_and_bed_exit() -> None:
    box = (1, 2, 21, 42, 0.9)
    observed: dict[str, tuple[int | None, ...]] = {}

    class _Tracker:
        @property
        def live_ids(self) -> frozenset[int]:
            return frozenset({42})

        def update(self, boxes):
            assert len(boxes) == 1
            return (42,)

    class _PoseRunner:
        calls = 0

        def run(self, image: np.ndarray):
            del image
            self.calls += 1
            keypoints = tuple((1.0, 2.0, 0.9) for _ in range(17))
            return pose_result((keypoints,), (box,))

    class _FallConsumer:
        def update(self, domain_input):
            observed["fall"] = domain_input.observation.track_ids
            assert domain_input.live_track_ids == (42,)
            return ()

    class _BedExitConsumer:
        def update(self, domain_input):
            observed["bed_exit"] = domain_input.observation.track_ids
            return ()

    pose_runner = _PoseRunner()

    worker = CameraWorker(
        camera_id="camera-1",
        facility_id="facility-1",
        frame_source=(),
        runners={"pose": pose_runner},

        scheduler=Scheduler({"pose": 1}),

        domain_detectors=(_FallConsumer(), _BedExitConsumer()),
        tracker=_Tracker(),
    )

    observation = worker.process_frame(
        Frame(index=0, time_sec=1.0, image=np.zeros((10, 10, 3), dtype=np.uint8))
    )

    assert observation.track_ids == (42,)
    assert observed == {"fall": (42,), "bed_exit": (42,)}
    assert observation.poses == (tuple((1.0, 2.0, 0.9) for _ in range(17)),)
    assert pose_runner.calls == 1


def _two_camera_config() -> EdgeWorkerConfig:
    return EdgeWorkerConfig(
        version=1,
        relay={"url": "http://127.0.0.1:8000", "token": "relay-token"},
        cameras=tuple(
            CameraRuntimeConfig(
                camera_id=f"camera-{index}",
                facility_id="facility-1",
                resident_id=f"resident-{index}",
                rtsp_url=f"rtsp://camera-{index}.local/trackID=2",
            )
            for index in range(1, 3)
        ),
    )
