from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts.frame import Frame
from contracts.observation import FrameObservation
from contracts.runner import RunnerOutput, bed_result, person_result
from edge.domains import DOMAIN_REGISTRY
from edge.domains.bed_exit.detector import BedExitMonitor
from edge.domains.bed_exit.schema import DomainDebugSnapshot
from edge.perception.domain_input import DomainInput
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.scheduler import Scheduler

BED = (0, 0, 100, 100, 0.9)
PERSON = (10, 10, 50, 50, 0.9)


class _Runner:
    def run(self, image: np.ndarray) -> RunnerOutput:
        del image
        return person_result((PERSON,))


class _BedRunner:
    def run(self, image: np.ndarray) -> RunnerOutput:
        del image
        return bed_result((BED,))


class _CountingMonitor(BedExitMonitor):
    def __init__(self) -> None:
        super().__init__(hold_frames=1, grace_frames=0)
        self.update_count = 0
        self.registration = DOMAIN_REGISTRY["bed_exit"]

    def update(self, domain_input: DomainInput):
        self.update_count += 1
        return super().update(domain_input)


@dataclass(slots=True)
class _OverlaySink:
    calls: list[tuple[str, int, FrameObservation, tuple[DomainDebugSnapshot, ...]]] = field(
        default_factory=list
    )

    def publish(
        self,
        camera_id: str,
        frame: Frame,
        observation: FrameObservation,
        debug_snapshots: tuple[DomainDebugSnapshot, ...],
    ) -> None:
        self.calls.append((camera_id, frame.index, observation, debug_snapshots))


def _frame(index: int = 0) -> Frame:
    return Frame(index=index, time_sec=float(index), image=np.zeros((120, 120, 3), dtype=np.uint8))


def test_overlay_debug_snapshot_is_from_single_detector_update_pass() -> None:
    monitor = _CountingMonitor()
    sink = _OverlaySink()
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {"person": _Runner(), "bed": _BedRunner()},
        scheduler=Scheduler({"person": 1, "bed": 1}),
        domain_detectors=(monitor,),
        overlay_sink=sink,
    )

    worker.process_frame(_frame(0))

    assert monitor.update_count == 1
    assert len(sink.calls) == 1
    _, frame_index, observation, snapshots = sink.calls[0]
    assert frame_index == 0
    assert observation.bed_boxes
    assert snapshots[0].bed_exit is not None
    assert snapshots[0].bed_exit.frame_index == 0
    assert snapshots[0].bed_exit.bed_region is not None
    assert snapshots[0].bed_exit.bed_region.source == "fresh"


def test_camera_worker_publishes_overlay_after_detector_update() -> None:
    monitor = _CountingMonitor()
    sink = _OverlaySink()
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {"person": _Runner(), "bed": _BedRunner()},
        scheduler=Scheduler({"person": 1, "bed": 1}),
        domain_detectors=(monitor,),
        overlay_sink=sink,
    )

    worker.process_frame(_frame(7))

    assert monitor.update_count == 1
    assert sink.calls[0][1] == 7
    assert sink.calls[0][3][0].bed_exit is not None
    assert sink.calls[0][3][0].bed_exit.frame_index == 7
