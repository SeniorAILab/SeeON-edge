from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts.event import EventPayload
from contracts.frame import Frame
from contracts.observation import BoundingBox
from contracts.runner import RunnerOutput, bed_result, person_result
from edge.domains.bed_exit.detector import BedExitMonitor
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.scheduler import Scheduler

BED = (0, 0, 100, 100, 0.9)
IN_BED = (10, 10, 50, 50, 0.9)
OUT_OF_BED = (150, 150, 190, 190, 0.9)


class _Runner:
    def __init__(self, outputs: dict[int, RunnerOutput] | RunnerOutput, kind: str) -> None:
        self.outputs = outputs
        self.kind = kind

    def run(self, image: np.ndarray) -> RunnerOutput:
        frame_index = int(image[0, 0, 0])
        output = (
            self.outputs.get(frame_index, ())
            if isinstance(self.outputs, dict)
            else self.outputs
        )
        return bed_result(output) if self.kind == "bed" else person_result(output)


@dataclass(slots=True)
class _Sink:
    events: list[EventPayload] = field(default_factory=list)

    def emit(self, event: EventPayload) -> None:
        self.events.append(event)


def _frame(index: int) -> Frame:
    return Frame(
        index=index, time_sec=float(index), image=np.full((4, 4, 3), index, dtype=np.uint8)
    )


def _worker(
    *,
    bed_interval: int,
    bed_outputs: dict[int, RunnerOutput],
    sink: _Sink,
    person_outputs: dict[int, RunnerOutput] | None = None,
) -> CameraWorker:
    people = person_outputs or {
        0: (IN_BED,),
        1: (IN_BED,),
        2: (OUT_OF_BED,),
        3: (OUT_OF_BED,),
        4: (OUT_OF_BED,),
    }
    return CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=(),
        runners={"person": _Runner(people, "person"), "bed": _Runner(bed_outputs, "bed")},
        scheduler=Scheduler({"person": 1, "bed": bed_interval}),
        domain_detectors=(
            BedExitMonitor(
                min_containment=0.8,
                hold_frames=1,
                grace_frames=0,
            ),
        ),
        event_sink=sink,
    )


def test_expired_cached_roi_cannot_fabricate_bed_exit() -> None:
    sink = _Sink()
    worker = _worker(bed_interval=1, bed_outputs={0: (BED,), 1: (), 2: ()}, sink=sink)
    worker.process_frame(_frame(0))
    worker.process_frame(_frame(1))
    worker.process_frame(_frame(2))
    assert sink.events == []
    assert worker.scene_state is not None
    assert worker.scene_state.bed_region_counters.expired >= 1


def test_expired_cached_roi_cannot_suppress_fresh_bed_exit() -> None:
    sink = _Sink()
    worker = _worker(
        bed_interval=1,
        bed_outputs={0: (BED,), 1: (), 2: (), 3: (BED,), 4: (BED,)},
        sink=sink,
        person_outputs={
            0: (IN_BED,),
            1: (IN_BED,),
            2: (OUT_OF_BED,),
            3: ((0, 0, 100, 100, 0.9),),
            4: ((30, 30, 130, 130, 0.9),),
        },
    )
    for index in range(5):
        worker.process_frame(_frame(index))
    assert [event["event_type"] for event in sink.events] == ["bed-exit"]


def test_camera_worker_constructs_scene_state_per_camera_and_preserves_shared_runners() -> None:
    runner = _Runner((), "person")
    first = CameraWorker("cam-1", "facility-1", (), {"person": runner})
    second = CameraWorker("cam-2", "facility-1", (), {"person": runner})
    assert first.runners["person"] is second.runners["person"]
    assert first.scene_state is not None
    assert second.scene_state is not None
    assert first.scene_state.camera_id == "cam-1"
    assert second.scene_state.camera_id == "cam-2"
    first.scene_state.resolve_bed_regions(
        first.observation_builder(bed_boxes=(BoundingBox(*BED),)),
        frame_index=0,
        bed_scheduled=True,
        bed_interval=30,
    )
    assert second.scene_state.bed_regions == ()


def test_cached_roi_freshness_counters_are_observable() -> None:
    sink = _Sink()
    worker = _worker(bed_interval=2, bed_outputs={0: (BED,)}, sink=sink)
    worker.process_frame(_frame(0))
    worker.process_frame(_frame(1))
    worker.process_frame(_frame(5))
    categories = [
        event.category
        for event in worker.status_store.ops_events()
        if event.event_type == "bed_roi.cache"
    ]
    assert "fresh" in categories
    assert "cached" in categories
    assert "expired" in categories
