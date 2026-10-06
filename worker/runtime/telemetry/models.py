from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from contracts.encode_diagnostics import EncodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.inference_telemetry import (
    CameraInferenceTelemetry,
    InferenceTelemetrySnapshot,
)
from worker.pipeline.perception.scene_state import BedRegionCacheCounterSnapshot


@dataclass(frozen=True, slots=True)
class BedRegionDiagnostics:
    freshness: BedRegionCacheState
    counters: BedRegionCacheCounterSnapshot
    updated_at_sec: float


@dataclass(frozen=True, slots=True)
class BedExitScoringDiagnostics:
    max_containment_observed: float
    grace_positive_transitions: int
    assignments_made: int
    updated_at_sec: float


@dataclass(frozen=True, slots=True)
class DecodeBackendObservability:
    requested_profile_decode: str
    resolved_backend: str
    actual_adapter_class: str


@dataclass(frozen=True, slots=True)
class StageTimingSnapshot:
    stage: str
    samples: int
    total_sec: float
    last_sec: float
    max_sec: float


@dataclass(frozen=True, slots=True)
class DeviceResidencyDiagnostics:
    residency_path: str
    h2d_transfers: int
    h2d_bytes: int
    d2h_transfers: int
    d2h_bytes: int
    pool_capacity: int
    pool_outstanding: int
    pool_high_watermark: int
    pool_exhaustion_events: int
    decode_time_ms_total: float
    decode_samples: int
    inference_time_ms_total: float
    inference_samples: int
    unavailable_reason: str | None
    updated_at_sec: float


@dataclass(frozen=True, slots=True)
class BusSubscriptionSnapshot:
    name: str
    published: int
    taken: int
    dropped: int
    queue_age_sec: float


@dataclass(frozen=True, slots=True)
class EncoderLifecycleSnapshot:
    process_starts: int = 0
    recreates: int = 0
    failures: int = 0
    active_sessions: int = 0
    finalized_segments: int = 0
    unavailable_cameras: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryBatchHistogram:
    geometry: tuple[int, int]
    batch_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CameraDiagnosticsSnapshot:
    camera_id: str
    failure_category: str | None
    stage_timings: tuple[StageTimingSnapshot, ...]
    bus: tuple[BusSubscriptionSnapshot, ...]
    decode_backend: DecodeBackendObservability | None = None
    encode: EncodeSelection | None = None
    bed_region: BedRegionDiagnostics | None = None
    bed_exit_scoring: BedExitScoringDiagnostics | None = None
    device_residency: DeviceResidencyDiagnostics | None = None
    decision_completed: int = 0
    inference: CameraInferenceTelemetry | None = None
    batch_sizes: tuple[tuple[int, int], ...] = ()
    geometry_batch_sizes: tuple[GeometryBatchHistogram, ...] = ()
    forward_p50_sec: float = 0.0
    forward_p95_sec: float = 0.0
    track_id_switch_total: int = 0
    replay_trace_write_failures: int = 0
    track_id_switch_absorbed_total: int = 0
    resample_gap_rows_total: int = 0
    incident_cooldown_suppressed_total: int = 0
    bed_polygon_source: str = "none"
    inference_fps: float | None = None
    camera_fps_unpinned: bool = False
    fall_inference_device: str = "unknown"
    fall_unapplied_policy_threshold: float | None = None
    smart_record_extended_total: int = 0
    smart_record_extension_raced_total: int = 0
    smart_record_start_refused_total: int = 0
    nvenc_sessions_active: int = 0
    flow_source_outages_total: int = 0
    flow_source_recoveries_total: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticsSnapshot:
    cameras: tuple[CameraDiagnosticsSnapshot, ...]
    encoder: EncoderLifecycleSnapshot


class SubscriptionMetrics(Protocol):
    @property
    def published(self) -> int: ...

    @property
    def taken(self) -> int: ...

    @property
    def dropped(self) -> int: ...

    @property
    def queue_age_sec(self) -> float: ...


class InferenceMetricsSource(Protocol):
    def snapshot(self) -> InferenceTelemetrySnapshot: ...


class BusMetricsSource(Protocol):
    def metrics(self, name: str) -> SubscriptionMetrics: ...


@final
class InvalidStageTimingError(ValueError):
    __slots__ = ("elapsed_sec",)

    def __init__(self, elapsed_sec: float) -> None:
        self.elapsed_sec = elapsed_sec
        super().__init__(f"stage timing must be non-negative: {elapsed_sec}")


__all__ = [
    "BedExitScoringDiagnostics",
    "BedRegionDiagnostics",
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "DecodeBackendObservability",
    "DeviceResidencyDiagnostics",
    "EncoderLifecycleSnapshot",
    "GeometryBatchHistogram",
    "InferenceMetricsSource",
    "InvalidStageTimingError",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
]
