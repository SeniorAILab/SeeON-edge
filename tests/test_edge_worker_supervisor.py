from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from contracts.frame import Frame, FrameSource
from contracts.observation import FrameObservation
from contracts.runner import RunnerOutput, bed_result, person_result
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker_supervisor import EdgeWorkerSupervisor
from edge.runtime.runtime_diagnostics import WorkerDiagnostics
from edge.runtime.scheduler import Scheduler
from edge.runtime.status_store import CameraStatus, StatusStore
from edge.sources.rtsp import RTSPSource


class _FiniteSource:
    def __init__(self, offset: int, count: int = 3) -> None:
        self.offset = offset
        self.count = count

    def __iter__(self) -> Iterator[Frame]:
        for index in range(self.count):
            image = np.full((1, 1, 3), self.offset + index, dtype=np.uint8)
            yield Frame(index=index, time_sec=float(index), image=image)


class _FailingSource:
    def __iter__(self) -> Iterator[Frame]:
        raise OSError("rtsp offline")
        yield


class _EmptySource:
    def __iter__(self) -> Iterator[Frame]:
        return
        yield


class _RecoveringSource:
    def __init__(self) -> None:
        self._on_reconnecting = None
        self._on_recovered = None

    def set_liveness_callbacks(self, *, on_reconnecting=None, on_recovered=None) -> None:
        self._on_reconnecting = on_reconnecting
        self._on_recovered = on_recovered

    def __iter__(self) -> Iterator[Frame]:
        if self._on_reconnecting is not None:
            self._on_reconnecting("read_failure")
        if self._on_recovered is not None:
            self._on_recovered("read_recovered")
        image = np.full((1, 1, 3), 10, dtype=np.uint8)
        yield Frame(index=0, time_sec=0.0, image=image)


class _ThreadRecordingRunner:
    def __init__(self) -> None:
        self.thread_names: list[str] = []
        self.values: list[int] = []

    def run(self, image: np.ndarray) -> RunnerOutput:
        self.thread_names.append(threading.current_thread().name)
        self.values.append(int(image[0, 0, 0]))
        return person_result(())


class _TwoBedRunner:
    def run(self, image: np.ndarray) -> RunnerOutput:
        del image
        return bed_result(((0, 0, 10, 10, 0.91), (20, 20, 30, 30, 0.92)))


class _ObservationRecorder:
    def __init__(self) -> None:
        self.observations: list[FrameObservation] = []

    def update(self, domain_input: object) -> tuple[object, ...]:
        self.observations.append(domain_input.observation)  # type: ignore[union-attr]
        return ()


class _HeartbeatSink:
    def __init__(self) -> None:
        self.calls = 0

    def send_heartbeat(self) -> bool:
        self.calls += 1
        return True


def test_one_offline_camera_does_not_stop_other_three() -> None:
    status_store = StatusStore()
    workers = [
        _worker("camera-1", _FiniteSource(10), status_store),
        _worker("camera-2", _FiniteSource(20), status_store),
        _worker("camera-3", _FailingSource(), status_store),
        _worker("camera-4", _FiniteSource(40), status_store),
    ]
    supervisor = EdgeWorkerSupervisor.from_workers(workers, status_store=status_store)

    result = supervisor.run(max_frames_per_camera=1)

    assert result == {"camera-1": 1, "camera-2": 1, "camera-3": 0, "camera-4": 1}
    assert _camera_status(status_store, "camera-3") == CameraStatus.DEGRADED
    assert _camera_status(status_store, "camera-1") == CameraStatus.READY
    assert _camera_status(status_store, "camera-2") == CameraStatus.READY
    assert _camera_status(status_store, "camera-4") == CameraStatus.READY


def test_heartbeats_are_sent_only_for_ready_cameras() -> None:
    status_store = StatusStore()
    ready_sink = _HeartbeatSink()
    degraded_sink = _HeartbeatSink()
    starting_sink = _HeartbeatSink()
    workers = [
        _worker("ready-camera", _FiniteSource(10), status_store),
        _worker("degraded-camera", _FiniteSource(20), status_store),
        _worker("starting-camera", _FiniteSource(30), status_store),
        _worker("unknown-camera", _FiniteSource(40), status_store),
    ]
    supervisor = EdgeWorkerSupervisor.from_workers(
        workers,
        status_store=status_store,
        heartbeat_sinks={
            "ready-camera": ready_sink,
            "degraded-camera": degraded_sink,
            "starting-camera": starting_sink,
        },
    )
    status_store.set_status("ready-camera", "facility-1", CameraStatus.READY)
    status_store.set_status("degraded-camera", "facility-1", CameraStatus.DEGRADED)
    status_store.set_status("starting-camera", "facility-1", CameraStatus.STARTING)

    supervisor._send_heartbeats()

    assert ready_sink.calls == 1
    assert degraded_sink.calls == 0
    assert starting_sink.calls == 0


def test_source_without_decoded_frames_never_marks_camera_ready() -> None:
    status_store = StatusStore()
    worker = _worker("camera-1", _EmptySource(), status_store)
    supervisor = EdgeWorkerSupervisor.from_workers((worker,), status_store=status_store)

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 0}
    assert _camera_status(status_store, "camera-1") == CameraStatus.STARTING


def test_rtsp_liveness_transitions_degraded_to_ready_on_recovery() -> None:
    status_store = StatusStore()
    worker = _worker("camera-1", _RecoveringSource(), status_store)
    supervisor = EdgeWorkerSupervisor.from_workers((worker,), status_store=status_store)

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}

    assert _camera_status(status_store, "camera-1") == CameraStatus.READY
    liveness_events = [
        (event.event_type, event.category)
        for event in status_store.ops_events()
        if event.event_type.startswith("camera.")
    ]
    assert liveness_events == [
        ("camera.offline", "rtsp_reconnecting"),
        ("camera.recovered", "rtsp_recovered"),
    ]

def test_supervisor_processing_updates_measured_fps() -> None:
    status_store = StatusStore()
    diagnostics = WorkerDiagnostics()
    diagnostics.register_decode("camera-1", "auto")
    worker = CameraWorker(
        camera_id="camera-1",
        facility_id="facility-1",
        frame_source=_FiniteSource(10, count=2),
        runners={},
        status_store=status_store,
        diagnostics=diagnostics,
    )

    supervisor = EdgeWorkerSupervisor.from_workers((worker,), status_store=status_store)
    supervisor._process_frame(worker, Frame(index=0, time_sec=0.0, image=np.zeros((1, 1, 3))))
    supervisor._process_frame(worker, Frame(index=1, time_sec=1.0, image=np.zeros((1, 1, 3))))

    camera = diagnostics.to_payload("facility-1", None, 1)["cameras"][0]
    assert camera["measured_fps"] is not None


def test_runner_called_by_scheduler_only() -> None:
    status_store = StatusStore()
    runner = _ThreadRecordingRunner()
    worker = CameraWorker(
        camera_id="camera-1",
        facility_id="facility-1",
        frame_source=_FiniteSource(10, count=2),
        runners={"pose": runner},
        scheduler=Scheduler({"pose": 1}),
        status_store=status_store,
    )
    supervisor = EdgeWorkerSupervisor.from_workers((worker,), status_store=status_store)

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}

    assert runner.values
    assert runner.thread_names
    assert all(not name.startswith("edge-capture") for name in runner.thread_names)


def test_two_bed_boxes_are_not_misread_as_pose_pair() -> None:
    detector = _ObservationRecorder()
    worker = CameraWorker(
        camera_id="camera-1",
        facility_id="facility-1",
        frame_source=_FiniteSource(10, count=1),
        runners={"bed": _TwoBedRunner()},
        scheduler=Scheduler({"bed": 1}),
        domain_detectors=(detector,),
    )

    worker.process_frame(Frame(index=0, time_sec=0.0, image=np.zeros((1, 1, 3))))

    assert len(detector.observations) == 1
    assert [box.confidence for box in detector.observations[0].bed_boxes] == [0.91, 0.92]


def test_supervisor_stop_terminates_reconnecting_capture_thread_promptly() -> None:
    status_store = StatusStore()
    stop_event = threading.Event()
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        max_failures=1,
        reconnect_initial_backoff_sec=30.0,
        reconnect_max_backoff_sec=30.0,
        max_total_reconnects=None,
        backoff_wait=stop_event.wait,
        stop_requested=stop_event.is_set,
        backend=_NeverRecoveringBackend(),
    )
    worker = _worker("camera-1", source, status_store)
    supervisor = EdgeWorkerSupervisor.from_workers(
        (worker,),
        status_store=status_store,
        stop_event=stop_event,
    )

    supervisor._start_capture_threads()
    thread = supervisor.loops[0].thread
    assert thread is not None
    deadline = time.monotonic() + 1.0
    while status_store.get_status("camera-1") is None and time.monotonic() < deadline:
        time.sleep(0.001)

    supervisor.stop()

    assert not thread.is_alive()
    assert supervisor.loops[0].done.is_set()
    assert _camera_status(status_store, "camera-1") == CameraStatus.DEGRADED


class _NeverRecoveringCapture:
    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        pass


class _NeverRecoveringBackend:
    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> _NeverRecoveringCapture:
        del url, open_timeout_ms, read_timeout_ms
        return _NeverRecoveringCapture()

    def read(self, capture: _NeverRecoveringCapture) -> tuple[bool, None]:
        return capture.read()

    def release(self, capture: _NeverRecoveringCapture) -> None:
        capture.release()

def _worker(camera_id: str, source: FrameSource, status_store: StatusStore) -> CameraWorker:
    return CameraWorker(
        camera_id=camera_id,
        facility_id="facility-1",
        frame_source=source,
        runners={},
        status_store=status_store,
    )


def _camera_status(status_store: StatusStore, camera_id: str) -> CameraStatus:
    status = status_store.get_status(camera_id)
    assert status is not None
    return status.status
