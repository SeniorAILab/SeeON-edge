from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from contracts.frame import Frame
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker_supervisor import EdgeWorkerSupervisor
from edge.runtime.status_store import StatusStore


class _ThreeFrameSource:
    def __init__(self, camera_offset: int) -> None:
        self.camera_offset = camera_offset

    def __iter__(self) -> Iterator[Frame]:
        for index in range(3):
            image = np.full((1, 1, 3), self.camera_offset + index, dtype=np.uint8)
            yield Frame(index=index, time_sec=float(index), image=image)


def test_four_stream_worker_processes_each_camera() -> None:
    status_store = StatusStore()
    workers = tuple(
        CameraWorker(
            camera_id=f"camera-{index}",
            facility_id="facility-1",
            frame_source=_ThreeFrameSource(index * 10),
            runners={},
            status_store=status_store,
        )
        for index in range(1, 5)
    )

    supervisor = EdgeWorkerSupervisor.from_workers(workers, status_store=status_store)

    assert supervisor.run(max_frames_per_camera=1) == {
        "camera-1": 1,
        "camera-2": 1,
        "camera-3": 1,
        "camera-4": 1,
    }
