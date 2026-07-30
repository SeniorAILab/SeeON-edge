from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from contracts.frame import Frame
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.latest_frame import LatestFrameBuffer
from edge.runtime.status_store import CameraStatus, StatusStore


class HeartbeatSink(Protocol):
    def send_heartbeat(self) -> bool: ...


@dataclass(slots=True)
class _CameraLoop:
    worker: CameraWorker
    buffer: LatestFrameBuffer
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


@dataclass(slots=True)
class EdgeWorkerSupervisor:
    loops: tuple[_CameraLoop, ...]
    status_store: StatusStore
    heartbeat_sinks: dict[str, HeartbeatSink] = field(default_factory=dict)
    heartbeat_interval_sec: float = 30.0
    stop_event: threading.Event = field(default_factory=threading.Event)
    restart_check: Callable[[], bool] | None = None
    config_refresh: Callable[[], None] | None = None

    @classmethod
    def from_workers(
        cls,
        workers: tuple[CameraWorker, ...] | list[CameraWorker],
        *,
        status_store: StatusStore,
        heartbeat_sinks: dict[str, HeartbeatSink] | None = None,
        heartbeat_interval_sec: float = 30.0,
        stop_event: threading.Event | None = None,
        restart_check: Callable[[], bool] | None = None,
        config_refresh: Callable[[], None] | None = None,
    ) -> EdgeWorkerSupervisor:
        return cls(
            loops=tuple(
                _CameraLoop(worker=worker, buffer=LatestFrameBuffer()) for worker in workers
            ),
            status_store=status_store,
            heartbeat_sinks={} if heartbeat_sinks is None else heartbeat_sinks,
            heartbeat_interval_sec=heartbeat_interval_sec,
            stop_event=threading.Event() if stop_event is None else stop_event,
            restart_check=restart_check,
            config_refresh=config_refresh,
        )

    def run(
        self,
        *,
        max_frames_per_camera: int | None = None,
        heartbeat_on_start: bool = False,
    ) -> dict[str, int]:
        counts = {loop.worker.camera_id: 0 for loop in self.loops}
        self._start_capture_threads()
        last_heartbeat = time.monotonic()
        if heartbeat_on_start:
            self._send_heartbeats()
        try:
            while not self._done(counts, max_frames_per_camera):
                made_progress = False
                for loop in self.loops:
                    camera_id = loop.worker.camera_id
                    if _camera_complete(counts[camera_id], max_frames_per_camera):
                        continue
                    frame = loop.buffer.take(timeout_sec=0.01)
                    if frame is None:
                        continue
                    self._process_frame(loop.worker, frame)
                    counts[camera_id] += 1
                    made_progress = True
                now = time.monotonic()
                if now - last_heartbeat >= self.heartbeat_interval_sec:
                    self._send_heartbeats()
                    if self.config_refresh is not None:
                        self.config_refresh()
                    if self.restart_check is not None and self.restart_check():
                        self.stop_event.set()
                        break
                    last_heartbeat = now
                if not made_progress and all(loop.done.is_set() for loop in self.loops):
                    break
            return counts
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        for loop in self.loops:
            if loop.thread is not None:
                loop.thread.join(timeout=1.0)

    def _start_capture_threads(self) -> None:
        for loop in self.loops:
            thread = threading.Thread(
                target=self._capture_loop,
                args=(loop,),
                name=f"edge-capture-{loop.worker.camera_id}",
                daemon=True,
            )
            loop.thread = thread
            thread.start()

    def _capture_loop(self, loop: _CameraLoop) -> None:
        worker = loop.worker
        self.status_store.set_status(worker.camera_id, worker.facility_id, CameraStatus.STARTING)
        try:
            worker._bind_source_liveness_callbacks()
            frame_iter = iter(worker.frame_source)
            saw_frame = False
            for frame in frame_iter:
                if self.stop_event.is_set():
                    break
                if not saw_frame:
                    self.status_store.set_status(
                        worker.camera_id,
                        worker.facility_id,
                        CameraStatus.READY,
                    )
                    saw_frame = True
                if not loop.buffer.put(frame, stop_event=self.stop_event):
                    break
        except Exception as exc:  # noqa: BLE001 - camera source failures soft-degrade one camera
            worker._mark_source_failure(exc)
        finally:
            loop.done.set()

    def _process_frame(self, worker: CameraWorker, frame: Frame) -> None:
        try:
            worker.process_frame(frame)
        except Exception as exc:  # noqa: BLE001 - processing failures are per-frame ops events
            worker._mark_processing_failure(exc)

    def _send_heartbeats(self) -> None:
        for loop in self.loops:
            status = self.status_store.get_status(loop.worker.camera_id)
            if status is None or status.status != CameraStatus.READY:
                continue
            sink = self.heartbeat_sinks.get(loop.worker.camera_id)
            if sink is not None:
                sink.send_heartbeat()

    def _done(self, counts: dict[str, int], max_frames_per_camera: int | None) -> bool:
        if max_frames_per_camera is None:
            return False
        return all(count >= max_frames_per_camera for count in counts.values())


def _camera_complete(count: int, max_frames_per_camera: int | None) -> bool:
    return max_frames_per_camera is not None and count >= max_frames_per_camera


__all__ = ["EdgeWorkerSupervisor", "HeartbeatSink"]
