"""Worker-local diagnostics and frozen runtime-status wire projection."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from time import monotonic, time
from typing import final

from contracts.decode_diagnostics import DECODE_FALLBACK_REASONS, DecodeSelection
from worker.runtime.telemetry.local_metrics import (
    StageTimingAccumulator,
    bus_snapshot,
    log_snapshot,
)
from worker.runtime.telemetry.models import (
    BusMetricsSource,
    BusSubscriptionSnapshot,
    CameraDiagnosticsSnapshot,
    EncoderLifecycleSnapshot,
    InvalidStageTimingError,
    RuntimeDiagnosticsSnapshot,
    StageTimingSnapshot,
    SubscriptionMetrics,
)
from worker.runtime.telemetry.status_store import StatusStore
from worker.runtime.telemetry.wire import (
    ClipRecorderStatus,
    RelayCameraPayload,
    RelayClipRecorderPayload,
    RelayDecodePayload,
    RelayGpuPayload,
    RelayRuntimeStatusPayload,
    RelayWorkerPayload,
    camera_payload,
    facility_payload,
)

MEASURED_FPS_MAX_AGE_SEC = 10.0


@final
class WorkerDiagnostics:
    """Thread-safe local metrics and strict legacy relay projection."""

    def __init__(
        self,
        status_store: StatusStore | None = None,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        self._lock = threading.RLock()
        self._status_store = StatusStore() if status_store is None else status_store
        self._clock = clock
        self._wall_clock = wall_clock
        self._decode_by_camera: dict[str, DecodeSelection] = {}
        self._measured_fps_by_camera: dict[str, tuple[float, float | None]] = {}
        self._stage_timings: dict[str, dict[str, StageTimingAccumulator]] = {}
        self._buses: dict[str, tuple[BusMetricsSource, tuple[str, ...]]] = {}
        self._encoder = EncoderLifecycleSnapshot()
        self._clip_recorder = ClipRecorderStatus()
        self._gpu: RelayGpuPayload | None = None
        self._worker: RelayWorkerPayload | None = None

    def set_clip_recorder_status(self, status: ClipRecorderStatus) -> None:
        with self._lock:
            self._clip_recorder = status

    def set_gpu_status(self, status: RelayGpuPayload | None) -> None:
        with self._lock:
            self._gpu = None if status is None else status.copy()

    def set_worker_status(self, status: RelayWorkerPayload | None) -> None:
        with self._lock:
            self._worker = None if status is None else status.copy()

    def register_decode(self, camera_id: str, requested: str) -> None:
        self.update_decode(
            camera_id,
            DecodeSelection(
                requested=requested,
                selected=None,
                fallback_count=0,
                last_reason=None,
                updated_at_sec=self._wall_clock(),
            ),
        )

    def update_decode(self, camera_id: str, selection: DecodeSelection) -> None:
        with self._lock:
            self._decode_by_camera[camera_id] = selection

    def record_decode_open_failure(self, camera_id: str, reason: str) -> None:
        normalized = reason if reason in DECODE_FALLBACK_REASONS else "spawn_failed"
        with self._lock:
            previous = self._decode_by_camera.get(camera_id)
            if previous is None:
                return
            self._decode_by_camera[camera_id] = DecodeSelection(
                requested=previous.requested,
                selected=None,
                fallback_count=previous.fallback_count,
                last_reason=normalized,
                updated_at_sec=self._wall_clock(),
            )

    def decode_selection(self, camera_id: str) -> DecodeSelection | None:
        with self._lock:
            return self._decode_by_camera.get(camera_id)

    def decode_snapshot(self) -> Mapping[str, DecodeSelection]:
        with self._lock:
            return dict(self._decode_by_camera)

    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        with self._lock:
            self._measured_fps_by_camera[camera_id] = (self._clock(), measured_fps)

    def record_stage_timing(self, camera_id: str, stage: str, elapsed_sec: float) -> None:
        if elapsed_sec < 0:
            raise InvalidStageTimingError(elapsed_sec)
        with self._lock:
            camera_stages = self._stage_timings.setdefault(camera_id, {})
            camera_stages.setdefault(stage, StageTimingAccumulator()).add(elapsed_sec)

    def register_bus(
        self,
        camera_id: str,
        bus: BusMetricsSource,
        subscriptions: tuple[str, ...] = ("inference", "live", "evidence"),
    ) -> None:
        with self._lock:
            self._buses[camera_id] = (bus, subscriptions)

    def update_encoder_lifecycle(self, snapshot: EncoderLifecycleSnapshot) -> None:
        with self._lock:
            self._encoder = EncoderLifecycleSnapshot(
                process_starts=snapshot.process_starts,
                recreates=snapshot.recreates,
                failures=snapshot.failures,
                active_sessions=snapshot.active_sessions,
                finalized_segments=snapshot.finalized_segments,
                unavailable_cameras=tuple(sorted(snapshot.unavailable_cameras)),
            )

    def snapshot(self) -> RuntimeDiagnosticsSnapshot:
        statuses = {
            record.camera_id: record for record in self._status_store.snapshot().cameras
        }
        with self._lock:
            stage_timings = {
                camera_id: tuple(
                    stages[stage].snapshot(stage) for stage in sorted(stages)
                )
                for camera_id, stages in self._stage_timings.items()
            }
            buses = dict(self._buses)
            encoder = self._encoder
            camera_ids = (
                set(self._decode_by_camera)
                | set(stage_timings)
                | set(buses)
                | set(statuses)
            )
        cameras = tuple(
            CameraDiagnosticsSnapshot(
                camera_id=camera_id,
                failure_category=(
                    None if camera_id not in statuses else statuses[camera_id].error_category
                ),
                stage_timings=stage_timings.get(camera_id, ()),
                bus=bus_snapshot(buses.get(camera_id)),
            )
            for camera_id in sorted(camera_ids)
        )
        return RuntimeDiagnosticsSnapshot(cameras=cameras, encoder=encoder)

    def log_snapshot(self) -> None:
        log_snapshot(self.snapshot())

    def to_payload(
        self,
        facility_id: str,
        generation: int | None,
        seq: int,
    ) -> RelayRuntimeStatusPayload:
        selections, measured_fps, clip_recorder, gpu, worker = self._wire_inputs()
        cameras = [
            camera_payload(camera_id, selection, measured_fps.get(camera_id))
            for camera_id, selection in sorted(selections.items())
        ]
        return facility_payload(
            facility_id,
            generation,
            seq,
            cameras,
            clip_recorder,
            gpu,
            worker,
        )

    def to_payloads(
        self,
        camera_facilities: Mapping[str, str],
        generation: int | None,
        seq: int,
    ) -> list[RelayRuntimeStatusPayload]:
        selections, measured_fps, clip_recorder, gpu, worker = self._wire_inputs()
        cameras_by_facility: dict[str, list[RelayCameraPayload]] = {
            facility_id: [] for facility_id in set(camera_facilities.values())
        }
        for camera_id, selection in selections.items():
            facility_id = camera_facilities.get(camera_id)
            if facility_id is not None:
                cameras_by_facility[facility_id].append(
                    camera_payload(camera_id, selection, measured_fps.get(camera_id))
                )
        return [
            facility_payload(
                facility_id,
                generation,
                seq,
                cameras,
                clip_recorder,
                gpu,
                worker,
            )
            for facility_id, cameras in sorted(cameras_by_facility.items())
        ]

    def _wire_inputs(
        self,
    ) -> tuple[
        dict[str, DecodeSelection],
        dict[str, float | None],
        ClipRecorderStatus,
        RelayGpuPayload | None,
        RelayWorkerPayload | None,
    ]:
        with self._lock:
            selections = dict(self._decode_by_camera)
            measured_fps = {
                camera_id: measured
                for camera_id, (updated_at, measured) in self._measured_fps_by_camera.items()
                if self._clock() - updated_at <= MEASURED_FPS_MAX_AGE_SEC
            }
            clip_recorder = self._clip_recorder
            gpu = None if self._gpu is None else self._gpu.copy()
            worker = None if self._worker is None else self._worker.copy()
        return selections, measured_fps, clip_recorder, gpu, worker
__all__ = [
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "ClipRecorderStatus",
    "EncoderLifecycleSnapshot",
    "RelayCameraPayload",
    "RelayClipRecorderPayload",
    "RelayDecodePayload",
    "RelayGpuPayload",
    "RelayRuntimeStatusPayload",
    "RelayWorkerPayload",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
    "WorkerDiagnostics",
]
