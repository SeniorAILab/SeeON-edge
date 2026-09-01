"""Transform recorded runtime samples into independently verifiable rung receipts."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from worker.tools.deepstream_canary.models import (
    ArtifactBindings,
    CameraSignals,
    CanaryMode,
    GpuSignals,
    LatencySignals,
    LiveProtectionSignals,
    NvdecSignals,
    RungReceipt,
    TimelineEntry,
    WorkloadFacts,
)
from worker.tools.deepstream_canary.report import write_once


class NativeWindowSample(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )

    schema_version: Literal[2]
    camera_id: str = Field(min_length=1)
    window_started_ns: int
    window_ended_ns: int
    decision_count: int = Field(ge=0)
    latency_samples_ms: tuple[float, ...]
    metadata_published: int = Field(ge=0)
    metadata_overwritten: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> NativeWindowSample:
        if self.window_ended_ns <= self.window_started_ns:
            raise ValueError("window span must be positive")
        return self


class CopyWindowSample(BaseModel):
    """One immutable per-camera child-copy telemetry window."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )

    schema_version: Literal[1]
    camera_id: str = Field(min_length=1)
    window_started_ns: int
    window_ended_ns: int
    frames: int = Field(ge=0)
    h2d_bytes_max: int = Field(ge=0)
    d2h_bytes_max: int = Field(ge=0)
    box_source: Literal["pose", "person"]
    pool_wait_us_p95: float = Field(ge=0)
    gpu_us_p95: float = Field(ge=0)
    surface_drops: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> CopyWindowSample:
        if self.window_ended_ns <= self.window_started_ns:
            raise ValueError("window span must be positive")
        return self


class RuntimeGpuSample(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    child_memory_mib: float = Field(ge=0)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    utilization: float = Field(ge=0, le=100)


class RecordedCameraTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )

    camera_id: str = Field(min_length=1)
    decision_window_counts: tuple[int, ...] = Field(min_length=1)
    decision_window_seconds: tuple[float, ...] = Field(min_length=1)
    telemetry_coverage_seconds: float = Field(default=0, ge=0)
    copy_window_frames: int = Field(ge=0)
    frame_window_spans_seconds: tuple[float, ...] = Field(min_length=1)
    h2d_bytes_max: int = Field(ge=0)
    d2h_bytes_max: int = Field(ge=0)
    box_source: Literal["pose", "person"]
    pool_wait_us_p95: float = Field(ge=0)
    gpu_us_p95: float = Field(ge=0)
    surface_drops: int = Field(ge=0)
    latency_samples_ms: tuple[float, ...] = Field(min_length=1)
    au_gaps: int = Field(ge=0)
    config_discontinuities: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)
    metadata_published: int = Field(gt=0)
    metadata_overwritten: int = Field(ge=0)
    event_evidence_parity: bool
    preview_ok: bool
    derivative_ok: bool

    @model_validator(mode="after")
    def _validate_windows(self) -> RecordedCameraTelemetry:
        if len(self.decision_window_counts) != len(self.decision_window_seconds):
            raise ValueError("decision window counts and durations must align")
        if any(count <= 0 for count in self.decision_window_counts):
            raise ValueError("decision window counts must be positive")
        if any(seconds <= 0 for seconds in self.decision_window_seconds):
            raise ValueError("decision window durations must be positive")
        if any(seconds <= 0 for seconds in self.frame_window_spans_seconds):
            raise ValueError("frame window spans must be positive")
        return self


class RecordedGpuTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    warmup_memory_mib: tuple[float, ...] = Field(min_length=1)
    steady_memory_mib: tuple[float, ...] = Field(min_length=1)
    recovery_memory_mib: tuple[float, ...] = Field(min_length=1)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    slack_samples_mib: tuple[float, ...] = Field(min_length=1)
    utilization_samples: tuple[float, ...] = Field(min_length=1)
    new_xids: tuple[int, ...]


class RecordedRungTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )

    schema_version: Literal[2]
    rung: str
    mode: CanaryMode
    camera_count: int = Field(ge=0)
    clean_steady_seconds: int = Field(ge=0)
    cameras: tuple[RecordedCameraTelemetry, ...]
    gpu: RecordedGpuTelemetry
    nvdec: NvdecSignals
    timeline: tuple[TimelineEntry, ...]
    live_protection: LiveProtectionSignals
    fault_windows: tuple[str, ...]
    workload: WorkloadFacts


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_rung_receipt(
    recorded: RecordedRungTelemetry,
    artifacts: ArtifactBindings,
) -> RungReceipt:
    """Compute all percentiles from raw samples rather than reported summaries."""
    cameras = tuple(
        CameraSignals(
            camera_id=camera.camera_id,
            fps_windows=tuple(
                count / seconds
                for count, seconds in zip(
                    camera.decision_window_counts,
                    camera.decision_window_seconds,
                    strict=True,
                )
            ),
            telemetry_coverage_seconds=camera.telemetry_coverage_seconds,
            copy_window_frames=camera.copy_window_frames,
            frame_window_spans_seconds=camera.frame_window_spans_seconds,
            h2d_bytes_max=camera.h2d_bytes_max,
            d2h_bytes_max=camera.d2h_bytes_max,
            box_source=camera.box_source,
            pool_wait_us_p95=camera.pool_wait_us_p95,
            gpu_us_p95=camera.gpu_us_p95,
            surface_drops=camera.surface_drops,
            latency_ms=LatencySignals(
                p50=_percentile(camera.latency_samples_ms, 0.50),
                p95=_percentile(camera.latency_samples_ms, 0.95),
                p99=_percentile(camera.latency_samples_ms, 0.99),
                max=max(camera.latency_samples_ms),
            ),
            au_gaps=camera.au_gaps,
            config_discontinuities=camera.config_discontinuities,
            timestamp_discontinuities=camera.timestamp_discontinuities,
            metadata_published=camera.metadata_published,
            metadata_overwritten=camera.metadata_overwritten,
            event_evidence_parity=camera.event_evidence_parity,
            preview_ok=camera.preview_ok,
            derivative_ok=camera.derivative_ok,
        )
        for camera in recorded.cameras
    )
    gpu = recorded.gpu
    return RungReceipt(
        schema_version=2,
        rung=recorded.rung,
        mode=recorded.mode,
        camera_count=recorded.camera_count,
        clean_steady_seconds=recorded.clean_steady_seconds,
        cameras=cameras,
        gpu=GpuSignals(
            child_pid=gpu.child_pid,
            warmup_peak_mib=max(gpu.warmup_memory_mib),
            steady_p50_mib=_percentile(gpu.steady_memory_mib, 0.50),
            steady_p95_mib=_percentile(gpu.steady_memory_mib, 0.95),
            recovery_mib=gpu.recovery_memory_mib[-1],
            global_used_mib=gpu.global_used_mib,
            total_mib=gpu.total_mib,
            minimum_slack_mib=min(gpu.slack_samples_mib),
            utilization_p50=_percentile(gpu.utilization_samples, 0.50),
            utilization_p95=_percentile(gpu.utilization_samples, 0.95),
            utilization_max=max(gpu.utilization_samples),
            new_xids=gpu.new_xids,
        ),
        nvdec=recorded.nvdec,
        timeline=recorded.timeline,
        live_protection=recorded.live_protection,
        fault_windows=recorded.fault_windows,
        workload=recorded.workload,
        artifacts=artifacts,
    )


def emit_rung_receipt(
    telemetry_path: Path,
    evidence_dir: Path,
    artifacts: ArtifactBindings,
) -> Path:
    recorded = RecordedRungTelemetry.model_validate_json(telemetry_path.read_bytes())
    receipt = build_rung_receipt(recorded, artifacts)
    destination = evidence_dir / "raw" / f"rung-{recorded.rung}.json"
    write_once(destination, (receipt.model_dump_json() + "\n").encode())
    return destination


__all__ = [
    "CopyWindowSample",
    "NativeWindowSample",
    "RecordedCameraTelemetry",
    "RecordedGpuTelemetry",
    "RecordedRungTelemetry",
    "RuntimeGpuSample",
    "build_rung_receipt",
    "emit_rung_receipt",
]
