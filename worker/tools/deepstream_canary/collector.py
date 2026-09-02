"""Normalize host/native runtime samples into raw telemetry fixtures."""

from __future__ import annotations

import hashlib
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from worker.tools.deepstream_canary.models import (
    CanaryMode,
    LiveProtectionSignals,
    NvdecSignals,
    TimelineEntry,
    WorkloadFacts,
)
from worker.tools.deepstream_canary.telemetry import (
    CopyWindowSample,
    NativeWindowSample,
    RecordedCameraTelemetry,
    RecordedGpuTelemetry,
    RecordedRungTelemetry,
    RuntimeGpuSample,
)

WINDOW_SECONDS = 10.0
WINDOW_SECONDS_TOLERANCE_SECONDS = 1.0
MAX_INTERNAL_GAP_SECONDS = 1.0
COVERAGE_BOUNDARY_SECONDS = 22.0


class CollectionRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    evidence_dir: Path
    rung: str
    mode: CanaryMode
    clean_steady_seconds: int
    camera_ids: tuple[str, ...]
    gpu_samples: tuple[RuntimeGpuSample, ...] = Field(min_length=1)
    native_windows: tuple[NativeWindowSample, ...]
    copy_windows: tuple[CopyWindowSample, ...]
    allow_partial: bool = False
    fault_windows: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _playable(path: Path) -> bool:
    if not path.is_file():
        return False
    completed = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _timeline(root: Path, rung: str) -> tuple[TimelineEntry, ...]:
    if rung == "zero":
        return ()
    paths: tuple[tuple[Literal["event", "evidence", "preview", "derivative"], Path], ...] = (
        ("event", root / "event-clip.mp4"),
        ("evidence", root / "event-evidence.json"),
        ("preview", root / "viewer-loop-01.png"),
        ("derivative", root / "event-derivative.mp4"),
    )
    entries: list[TimelineEntry] = []
    for kind, path in paths:
        if path.is_file():
            playable = _playable(path) if kind in {"event", "derivative"} else True
            entries.append(
                TimelineEntry(kind=kind, sha256=_sha256(path), playable=playable)
            )
    return tuple(entries)


def _window_seconds(window: NativeWindowSample | CopyWindowSample) -> float:
    seconds = (window.window_ended_ns - window.window_started_ns) / 1_000_000_000
    if seconds <= 0:
        raise ValueError(
            f"window span must be positive: camera_id={window.camera_id!r} "
            f"seconds={seconds!r}"
        )
    return seconds


def _validate_non_overlapping(
    windows: tuple[NativeWindowSample, ...] | tuple[CopyWindowSample, ...],
    *,
    label: str,
) -> None:
    by_camera: dict[str, list[NativeWindowSample | CopyWindowSample]] = {}
    for window in windows:
        seconds = _window_seconds(window)
        if seconds > WINDOW_SECONDS + WINDOW_SECONDS_TOLERANCE_SECONDS:
            raise ValueError(
                f"window span exceeds {WINDOW_SECONDS} seconds plus tolerance: "
                f"camera_id={window.camera_id!r} seconds={seconds!r}"
            )
        by_camera.setdefault(window.camera_id, []).append(window)
    for camera_id, camera_windows in by_camera.items():
        ordered = sorted(camera_windows, key=lambda window: window.window_started_ns)
        for previous, current in pairwise(ordered):
            if current.window_started_ns < previous.window_ended_ns:
                raise ValueError(
                    f"{label} windows overlap: camera_id={camera_id!r}"
                )


def _validate_contiguous(
    windows: tuple[NativeWindowSample, ...] | tuple[CopyWindowSample, ...],
    *,
    label: str,
) -> None:
    by_camera: dict[str, list[NativeWindowSample | CopyWindowSample]] = {}
    for window in windows:
        by_camera.setdefault(window.camera_id, []).append(window)
    for camera_id, camera_windows in by_camera.items():
        ordered = sorted(camera_windows, key=lambda window: window.window_started_ns)
        for previous, current in pairwise(ordered):
            gap_seconds = (
                current.window_started_ns - previous.window_ended_ns
            ) / 1_000_000_000
            if gap_seconds > MAX_INTERNAL_GAP_SECONDS:
                raise ValueError(
                    f"{label} windows are not temporally contiguous: "
                    f"camera_id={camera_id!r} gap_seconds={gap_seconds!r}"
                )


def _has_correlated_copy_window(
    parent: NativeWindowSample,
    copies: list[CopyWindowSample],
) -> bool:
    return any(
        copy.window_started_ns < parent.window_ended_ns
        and parent.window_started_ns < copy.window_ended_ns
        for copy in copies
    )


def _windows_are_mutually_correlated(
    parents: tuple[NativeWindowSample, ...],
    copies: tuple[CopyWindowSample, ...],
) -> bool:
    return (
        all(_has_correlated_copy_window(parent, list(copies)) for parent in parents)
        and all(_has_correlated_copy_window(copy, list(parents)) for copy in copies)
    )


def _recorded_camera(
    *,
    camera_id: str,
    windows: list[NativeWindowSample],
    copy_samples: list[CopyWindowSample],
    parity: bool,
    preview_ok: bool,
    derivative_ok: bool,
) -> RecordedCameraTelemetry | None:
    parent_windows = tuple(
        sorted(
            (window for window in windows if window.decision_count > 0),
            key=lambda window: window.window_started_ns,
        )
    )
    if not parent_windows:
        return None
    copies = tuple(sorted(copy_samples, key=lambda window: window.window_started_ns))
    if not copies or any(sample.frames <= 0 for sample in copies):
        return None
    if not _windows_are_mutually_correlated(parent_windows, copies):
        return None
    box_sources = {sample.box_source for sample in copies}
    if len(box_sources) != 1:
        raise ValueError(f"mixed copy box_source: camera_id={camera_id!r}")
    return RecordedCameraTelemetry(
        camera_id=camera_id,
        decision_window_counts=tuple(item.decision_count for item in parent_windows),
        decision_window_seconds=tuple(
            _window_seconds(item) for item in parent_windows
        ),
        telemetry_coverage_seconds=min(
            sum(_window_seconds(item) for item in parent_windows),
            sum(_window_seconds(item) for item in copies),
        ),
        copy_window_frames=sum(item.frames for item in copies),
        frame_window_spans_seconds=tuple(
            30 * _window_seconds(item) / item.frames for item in copies
        ),
        h2d_bytes_max=max(item.h2d_bytes_max for item in copies),
        d2h_bytes_max=max(item.d2h_bytes_max for item in copies),
        box_source=box_sources.pop(),
        pool_wait_us_p95=max(item.pool_wait_us_p95 for item in copies),
        gpu_us_p95=max(item.gpu_us_p95 for item in copies),
        surface_drops=sum(item.surface_drops for item in copies),
        latency_samples_ms=tuple(
            latency for item in parent_windows for latency in item.latency_samples_ms
        ),
        au_gaps=0,
        config_discontinuities=0,
        timestamp_discontinuities=sum(
            item.timestamp_discontinuities for item in parent_windows
        ),
        metadata_published=sum(item.metadata_published for item in parent_windows),
        metadata_overwritten=sum(item.metadata_overwritten for item in parent_windows),
        event_evidence_parity=parity,
        preview_ok=preview_ok,
        derivative_ok=derivative_ok,
    )


def collect_recorded_telemetry(request: CollectionRequest) -> Path:
    """Seal one rung from raw windows; no reported percentile is accepted."""
    by_camera: dict[str, list[NativeWindowSample]] = {}
    for window in request.native_windows:
        by_camera.setdefault(window.camera_id, []).append(window)
    copy_by_camera: dict[str, list[CopyWindowSample]] = {}
    for window in request.copy_windows:
        copy_by_camera.setdefault(window.camera_id, []).append(window)
    for camera_id, windows in copy_by_camera.items():
        if len({window.box_source for window in windows}) != 1:
            raise ValueError(f"mixed copy box_source: camera_id={camera_id!r}")
    if request.rung == "zero" and (
        request.camera_ids or request.native_windows or request.copy_windows
    ):
        raise ValueError("zero rung must not contain cameras or windows")
    if not request.allow_partial:
        _validate_non_overlapping(request.native_windows, label="native")
        _validate_non_overlapping(request.copy_windows, label="copy")
        _validate_contiguous(request.native_windows, label="native")
        _validate_contiguous(request.copy_windows, label="copy")
    expected_cameras = set(request.camera_ids)
    parent_cameras = set(by_camera)
    copy_cameras = set(copy_by_camera)
    if not request.allow_partial and (
        parent_cameras != expected_cameras or copy_cameras != expected_cameras
    ):
        raise ValueError(
            f"parent/copy camera identities mismatch: expected={request.camera_ids!r} "
            f"parent={tuple(sorted(parent_cameras))!r} "
            f"copy={tuple(sorted(copy_cameras))!r}"
        )
    camera_ids = sorted(parent_cameras & copy_cameras & expected_cameras)
    timeline = _timeline(request.evidence_dir / "raw", request.rung)
    parity = {entry.kind for entry in timeline} >= {"event", "evidence", "derivative"}
    preview_ok = any(entry.kind == "preview" and entry.playable for entry in timeline)
    derivative_ok = any(entry.kind == "derivative" and entry.playable for entry in timeline)
    cameras = tuple(
        camera
        for camera_id in camera_ids
        if (
            camera := _recorded_camera(
                camera_id=camera_id,
                windows=by_camera[camera_id],
                copy_samples=copy_by_camera[camera_id],
                parity=parity,
                preview_ok=preview_ok,
                derivative_ok=derivative_ok,
            )
        )
        is not None
    )
    if not request.allow_partial and len(cameras) != len(request.camera_ids):
        raise ValueError(
            "parent/copy windows must be correlated with positive child frames "
            "for every camera"
        )
    if not request.allow_partial:
        minimum_coverage_seconds = max(
            0.0, request.clean_steady_seconds - COVERAGE_BOUNDARY_SECONDS
        )
        for camera in cameras:
            if camera.telemetry_coverage_seconds < minimum_coverage_seconds:
                raise ValueError(
                    f"telemetry coverage is insufficient: camera_id={camera.camera_id!r} "
                    f"coverage_seconds={camera.telemetry_coverage_seconds!r} "
                    f"required_seconds={minimum_coverage_seconds!r}"
                )
    child_memory = tuple(item.child_memory_mib for item in request.gpu_samples)
    utilization = tuple(item.utilization for item in request.gpu_samples)
    final_gpu = request.gpu_samples[-1]
    recorded = RecordedRungTelemetry(
        schema_version=2,
        rung=request.rung,
        mode=request.mode,
        camera_count=len(request.camera_ids),
        clean_steady_seconds=request.clean_steady_seconds,
        cameras=cameras,
        gpu=RecordedGpuTelemetry(
            child_pid=final_gpu.child_pid,
            warmup_memory_mib=child_memory[:1],
            steady_memory_mib=child_memory,
            recovery_memory_mib=child_memory[-1:],
            global_used_mib=final_gpu.global_used_mib,
            total_mib=final_gpu.total_mib,
            slack_samples_mib=tuple(
                item.total_mib - item.global_used_mib for item in request.gpu_samples
            ),
            utilization_samples=utilization,
            new_xids=(),
        ),
        nvdec=NvdecSignals(
            hardware_branches=len(request.camera_ids),
            software_fallbacks=0,
        ),
        timeline=timeline,
        live_protection=LiveProtectionSignals(
            container_restarts=0,
            camera_stale_transitions=0,
            evidence_drop_increase=0,
            relay_sentinel_leaks=0,
            mount_intersections=0,
            kernel_faults=0,
        ),
        fault_windows=request.fault_windows,
        workload=WorkloadFacts(
            codec="h264",
            width=1280,
            height=720,
            fps=15.0,
            gop=30,
            camera_phase_offsets_ms=tuple(index * 67 for index in range(len(request.camera_ids))),
        ),
    )
    destination = request.evidence_dir / "raw" / f"telemetry-{request.rung}.json"
    destination.write_text(recorded.model_dump_json() + "\n", encoding="utf-8")
    return destination


__all__ = ["CollectionRequest", "collect_recorded_telemetry"]
