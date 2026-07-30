from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from contracts.decode_diagnostics import DECODE_FALLBACK_REASONS, DecodeSelection
from edge.evidence.clip_recorder import ClipRecorderStats


class WorkerDiagnostics:
    """Thread-safe worker telemetry snapshot source.

    Decode state is per camera. Clip recorder state is a single worker aggregate
    because the recorder owns one shared queue and encoder.
    """

    def __init__(
        self,
        clip_recorder_stats: ClipRecorderStats | None = None,
        *,
        gpu: Mapping[str, object] | None = None,
        worker: Mapping[str, object] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._decode_by_camera: dict[str, DecodeSelection] = {}
        self._clip_recorder_stats = clip_recorder_stats
        self._gpu = None if gpu is None else dict(gpu)
        self._worker = None if worker is None else dict(worker)
        self._measured_fps_by_camera: dict[str, tuple[float, float | None]] = {}

    def set_clip_recorder_stats(self, stats: ClipRecorderStats | None) -> None:
        with self._lock:
            self._clip_recorder_stats = stats

    def update_decode(self, camera_id: str, selection: DecodeSelection) -> None:
        with self._lock:
            self._decode_by_camera[camera_id] = selection
    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        with self._lock:
            self._measured_fps_by_camera[camera_id] = (time.monotonic(), measured_fps)
    def register_decode(self, camera_id: str, requested: str) -> None:
        with self._lock:
            self._decode_by_camera[camera_id] = DecodeSelection(
                requested=requested,
                selected=None,
                fallback_count=0,
                last_reason=None,
                updated_at_sec=time.time(),
            )

    def record_decode_open_failure(self, camera_id: str, reason: str) -> None:
        if reason not in DECODE_FALLBACK_REASONS:
            reason = "spawn_failed"
        with self._lock:
            previous = self._decode_by_camera.get(camera_id)
            if previous is None:
                return
            self._decode_by_camera[camera_id] = DecodeSelection(
                requested=previous.requested,
                selected=None,
                fallback_count=previous.fallback_count,
                last_reason=reason,
                updated_at_sec=time.time(),
            )

    def decode_selection(self, camera_id: str) -> DecodeSelection | None:
        with self._lock:
            return self._decode_by_camera.get(camera_id)

    def decode_snapshot(self) -> Mapping[str, DecodeSelection]:
        with self._lock:
            return dict(self._decode_by_camera)

    def to_payload(
        self,
        facility_id: str,
        generation: int | None,
        seq: int,
    ) -> dict[str, object]:
        with self._lock:
            selections = dict(self._decode_by_camera)
            stats = self._clip_recorder_stats
            gpu = None if self._gpu is None else dict(self._gpu)
            worker = None if self._worker is None else dict(self._worker)
            measured_fps = _fresh_measured_fps(self._measured_fps_by_camera)
        cameras = [
            _camera_payload(camera_id, selection, measured_fps.get(camera_id))
            for camera_id, selection in sorted(selections.items())
        ]
        return _facility_payload(facility_id, generation, seq, cameras, stats, gpu, worker)
    def to_payloads(
        self,
        camera_facilities: Mapping[str, str],
        generation: int | None,
        seq: int,
    ) -> list[dict[str, object]]:
        with self._lock:
            selections = dict(self._decode_by_camera)
            stats = self._clip_recorder_stats
            gpu = None if self._gpu is None else dict(self._gpu)
            worker = None if self._worker is None else dict(self._worker)
            measured_fps = _fresh_measured_fps(self._measured_fps_by_camera)
        cameras_by_facility: dict[str, list[dict[str, object]]] = {
            facility_id: [] for facility_id in set(camera_facilities.values())
        }
        for camera_id, selection in selections.items():
            facility_id = camera_facilities.get(camera_id)
            if facility_id is None:
                continue
            cameras_by_facility[facility_id].append(
                _camera_payload(camera_id, selection, measured_fps.get(camera_id))
            )
        return [
            _facility_payload(facility_id, generation, seq, cameras, stats, gpu, worker)
            for facility_id, cameras in sorted(cameras_by_facility.items())
        ]


def _fresh_measured_fps(
    values: Mapping[str, tuple[float, float | None]],
    *,
    now: float | None = None,
) -> dict[str, float | None]:
    current = time.monotonic() if now is None else now
    return {
        camera_id: measured
        for camera_id, (updated_at, measured) in values.items()
        if current - updated_at <= 10.0
    }

def _camera_payload(
    camera_id: str, selection: DecodeSelection, measured_fps: float | None
) -> dict[str, object]:
    payload: dict[str, object] = {"camera_id": camera_id, "decode": _decode_payload(selection)}
    if measured_fps is not None:
        payload["measured_fps"] = measured_fps
    return payload


def _facility_payload(
    facility_id: str,
    generation: int | None,
    seq: int,
    cameras: list[dict[str, object]],
    stats: ClipRecorderStats | None,
    gpu: dict[str, object] | None,
    worker: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "facility_id": facility_id,
        "generation": generation,
        "seq": seq,
        "cameras": sorted(cameras, key=lambda camera: str(camera["camera_id"])),
        "clip_recorder": _clip_payload(stats),
    }
    if gpu is not None:
        payload["gpu"] = gpu
    if worker is not None:
        payload["worker"] = worker
    return payload

def _decode_payload(selection: DecodeSelection) -> dict[str, object]:
    return {
        "requested": selection.requested,
        "selected": selection.selected,
        "fallback_count": selection.fallback_count,
        "last_reason": selection.last_reason,
        "updated_at_sec": selection.updated_at_sec,
    }


def _clip_payload(stats: ClipRecorderStats | None) -> dict[str, Any]:
    if stats is None:
        return {
            "available": False,
            "dropped_frames": None,
            "dropped_events": None,
            "failed_writes": None,
            "finalized_clips": None,
            "video_unavailable_clips": None,
            "active_clips": None,
            "encoder": None,
        }
    return {
        "available": True,
        "dropped_frames": stats.dropped_frames,
        "dropped_events": stats.dropped_events,
        "failed_writes": stats.failed_writes,
        "finalized_clips": stats.finalized_clips,
        "video_unavailable_clips": stats.video_unavailable_clips,
        "active_clips": getattr(stats, "active_clips", 0),
        "encoder": getattr(stats, "encoder", None),
    }


__all__ = ["WorkerDiagnostics"]
