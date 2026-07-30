from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from contracts.frame import Frame
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker_supervisor import EdgeWorkerSupervisor
from edge.runtime.status_store import StatusStore


class _FiniteSource:
    def __init__(self, count: int = 0) -> None:
        self.count = count

    def __iter__(self) -> Iterator[Frame]:
        for index in range(self.count):
            yield Frame(
                index=index,
                time_sec=float(index),
                image=np.zeros((1, 1, 3), dtype=np.uint8),
            )


def test_supervisor_restart_check_true_sets_stop_event_and_breaks_cleanly() -> None:
    status_store = StatusStore()
    calls = 0

    def restart_check() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    supervisor = EdgeWorkerSupervisor.from_workers(
        (_worker("camera-1", _FiniteSource()),),
        status_store=status_store,
        heartbeat_interval_sec=0.0,
        restart_check=restart_check,
    )

    assert supervisor.run() == {"camera-1": 0}
    assert supervisor.stop_event.is_set()
    assert calls == 1


def test_supervisor_restart_check_none_preserves_default_clean_completion() -> None:
    status_store = StatusStore()
    supervisor = EdgeWorkerSupervisor.from_workers(
        (_worker("camera-1", _FiniteSource(count=1)),),
        status_store=status_store,
        heartbeat_interval_sec=0.0,
    )

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}


def test_supervisor_restart_check_false_never_restarts() -> None:
    status_store = StatusStore()
    calls = 0

    def restart_check() -> bool:
        nonlocal calls
        calls += 1
        return False

    supervisor = EdgeWorkerSupervisor.from_workers(
        (_worker("camera-1", _FiniteSource(count=1)),),
        status_store=status_store,
        heartbeat_interval_sec=0.0,
        restart_check=restart_check,
    )

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}
    assert calls >= 1


def _worker(camera_id: str, source: _FiniteSource) -> CameraWorker:
    return CameraWorker(
        camera_id=camera_id,
        facility_id="facility-1",
        frame_source=source,
        runners={},
    )
