from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from time import monotonic
from types import MappingProxyType
from typing import Any, Final, Protocol, final, runtime_checkable

import worker.runtime.telemetry.runtime_status_sender as runtime_status_sender_module
from contracts.observation import BoundingBox
from contracts.runner import RunnerProtocol
from shared.detection_policies import LATEST_POLICY_VERSIONS
from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
)
from shared.events.relay_failure_log import RelayFailureLog
from shared.events.schemas import build_audit_envelope
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION
from worker.adapters.deepstream.service_maker import DeepStreamFlowStopTimeout
from worker.adapters.device.cuda.probe import probe_cuda_capability
from worker.adapters.device.mps.probe import probe_mps_capability
from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.adapters.media.ffmpeg_thumbnail import FfmpegThumbnailGenerator
from worker.adapters.model import ort_pose_bbox56, warmup_to_ready
from worker.adapters.model.clip_reanalysis import ClipAnalysisProfile
from worker.adapters.model.errors import FatalAcceleratorError, ModelLoadError
from worker.adapters.model.ort_bed_seg import BED_MODEL_CONFIDENCE, BED_ONNX_MODEL_PATH
from worker.domains import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_REGISTRY,
    DOMAIN_REGISTRY,
    CameraModuleContext,
    CompiledDetectionModuleRegistry,
    DetectionModuleDefinition,
    SharedComponentIdentity,
)
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallDomainDecider
from worker.domains.fall.classifier import FALL_STRIDE_FRAMES, FALL_WINDOW_FRAMES
from worker.domains.fall.pose_bbox56 import (
    COCO17_KEYPOINT_ORDER,
    POSE_BBOX56_CONFIDENCE_GATE,
)
from worker.domains.tracker import GreedyIouTracker
from worker.interfaces.clip_analysis import ClipAnalysisDisabledError
from worker.interfaces.clip_analysis import ClipAnalysisSupervisor as ClipAnalysisControl
from worker.interfaces.decision import (
    Decider,
    FreshnessProvider,
    ShadowTraceProvider,
    TraceSnapshotProvider,
)
from worker.interfaces.fall_model import FallModelProtocol
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics.merge import result_merger_names
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.output.evidence.clip_config import DEFAULT_CLIP_STORE_DIR
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import ClipPublisher, ReadyClipPublication
from worker.pipeline.output.evidence.clip_store_lock import (
    ClipStoreLock,
    ClipStoreLockedError,
)
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import (
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_config,
    start_optional_mjpeg_server,
)
from worker.pipeline.output.preview_renderer import PreviewRenderer
from worker.pipeline.perception import SceneState
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.runtime import bootstrap
from worker.runtime.clip_analysis_catchup import start_clip_analysis_catchup
from worker.runtime.clip_analysis_supervisor import ClipAnalysisSupervisor
from worker.runtime.config import (
    RELAY_HEARTBEAT_PATH,
    CameraRuntimeConfig,
    LiveClipExportPolicy,
    WorkerConfig,
    WorkerModelsConfig,
    replay_trace_directory_from_environment,
)
from worker.runtime.execution_records import compose_execution_records
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE, FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.flow.cold_start import FlowWarmupTimeout, verify_flow_boot_inputs
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.runtime.flow.lifecycle_supervisor import FlowLifecycleSupervisor
from worker.runtime.flow.media_plane import (
    BedZoneGeometry,
    FlowMediaPlane,
    FlowMediaPlaneConfig,
)
from worker.runtime.flow.policy_pump import (
    NativePolicyContext,
    NativePolicyPump,
)
from worker.runtime.lease import GpuLease
from worker.runtime.model_composition import SharedComponentGraph
from worker.runtime.nvidia_bed_zone_recognizer import (
    DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S,
    NvidiaBedZoneRecognizer,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    VerifyResult,
    default_verifiers,
)
from worker.runtime.provenance import (
    AppliedDetectionWindow,
    AppliedRuntimeManifest,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.environment import (
    RuntimeEnvironmentFacts,
    collect_runtime_environment_facts,
)
from worker.runtime.provenance.model_bundle import ModelBundleProof, admit_model_bundle
from worker.runtime.provenance.store import AppliedRuntimeManifestStore
from worker.runtime.state_dir import resolve_state_dir
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)
from worker.runtime.telemetry.wire import (
    RelayGpuPayload,
    RelayWorkerPayload,
)
from worker.runtime.watchdog import InferenceWatchdog
from worker.types import (
    CURRENT_TEMPORAL_PROFILE,
    BusinessEvent,
    DecisionInput,
    DecisionTraceSnapshot,
    TemporalProfile,
)
from worker.types.preview import FallPreviewState
from worker.types.trace import DecisionIdentity

LOGGER: Final = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT_SEC: Final = 6.0
DETECTOR_VERSION: Final = "worker-domain-detectors-v1"
CLIP_ANALYSIS_CPU_ENV: Final = "ML_WORKER_CLIP_ANALYSIS_CPU"


def _validate_fall_bundle_conformance(
    conformance: ort_pose_bbox56.PoseBbox56Conformance,
) -> None:
    if conformance.keypoint_order != COCO17_KEYPOINT_ORDER:
        raise ModelLoadError(
            "bundle conformance keypoint_order differs from COCO-17 domain order: "
            f"bundle {list(conformance.keypoint_order)!r}, "
            f"runner {list(COCO17_KEYPOINT_ORDER)!r}"
        )
    if conformance.confidence_gate != POSE_BBOX56_CONFIDENCE_GATE:
        raise ModelLoadError(
            "bundle conformance confidence.gate differs from domain contract: "
            f"bundle {conformance.confidence_gate:g}, "
            f"runner {POSE_BBOX56_CONFIDENCE_GATE:g}"
        )
    coordinate_system = conformance.coordinate_system
    expected_coordinates = {
        "origin": "top_left",
        "xy_normalization_denominators": {
            "x": "frame_width",
            "y": "frame_height",
        },
        "xy_normalization_rule": (
            "clip finite raw coordinates to inclusive raw bounds, then divide "
            "x by frame_width and y by frame_height"
        ),
    }
    observed_coordinates = {key: coordinate_system.get(key) for key in expected_coordinates}
    if observed_coordinates != expected_coordinates:
        raise ModelLoadError(
            "bundle conformance coordinate normalization differs from pose_bbox56 domain "
            f"contract: bundle {observed_coordinates!r}, runner {expected_coordinates!r}"
        )
    if conformance.window_frames != FALL_WINDOW_FRAMES:
        raise ModelLoadError(
            "bundle conformance temporal.window_frames differs from domain contract: "
            f"bundle {conformance.window_frames}, runner {FALL_WINDOW_FRAMES}"
        )
    expected_fps = CURRENT_TEMPORAL_PROFILE.pose_fps
    if conformance.stride_frames != FALL_STRIDE_FRAMES or conformance.fps != expected_fps:
        raise ModelLoadError(
            "bundle conformance temporal stride/fps differs from domain temporal profile: "
            f"bundle stride_frames={conformance.stride_frames}, fps={conformance.fps:g}; "
            f"runner stride_frames={FALL_STRIDE_FRAMES}, fps={expected_fps:g}"
        )


class EvidenceDeliveryError(RuntimeError):
    ...


def _persisted_bed_regions(camera: CameraRuntimeConfig) -> tuple[BoundingBox, ...]:
    return tuple(
        BoundingBox(
            x1=min(x for x, _ in region.polygon),
            y1=min(y for _, y in region.polygon),
            x2=max(x for x, _ in region.polygon),
            y2=max(y for _, y in region.polygon),
            confidence=1.0,
            polygon=region.polygon,
        )
        for region in camera.bed_zone_regions
    )


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


def _debug_snapshots_provider(
    domain_deciders: Mapping[str, Decider],
    definitions: Mapping[str, DetectionModuleDefinition] | None = None,
) -> Callable[[int], tuple[Any, ...]]:
    def provider(frame_index: int) -> tuple[Any, ...]:
        snapshots: list[Any] = []
        for name, detector in domain_deciders.items():
            adapter = (
                DOMAIN_REGISTRY[name].debug_snapshot_adapter
                if definitions is None
                else definitions[name].debug_adapter
            )
            if adapter is None:
                continue
            snapshot = adapter(detector, frame_index)
            if snapshot is not None:
                snapshots.append(snapshot)
        return tuple(snapshots)

    return provider


@dataclass(frozen=True, slots=True)
class _NativeEngineComponent:
    artifact_digest: str
    preprocessing_identity: str


@dataclass(frozen=True, slots=True)
class CameraDetectionPlan:
    tracker: GreedyIouTracker
    schedule: Mapping[str, int]
    detection_windows: Mapping[str, DetectionWindow | None]
    decision: EventAggregator
    domain_audit: Mapping[str, Mapping[str, object]]
    domain_deciders: Mapping[str, Decider]
    definitions: Mapping[str, DetectionModuleDefinition]


@final
class HeartbeatReporter:
    def __init__(self, worker: WorkerConfig, camera: CameraRuntimeConfig) -> None:
        self._worker, self._camera = worker, camera
        self._last_attempt: float | None = None
        self.failure_count = 0
        self._failure_log = RelayFailureLog(
            LOGGER, channel=f"heartbeat camera_id={camera.camera_id}", method="POST"
        )

    def mark_starting(self, camera_id: str) -> None:
        del camera_id

    def mark_ready(self, camera_id: str) -> None:
        now = monotonic()
        worker, camera = self._worker, self._camera
        if self._last_attempt is not None and (
            now < self._last_attempt + camera.heartbeat_interval_sec
        ):
            return
        self._last_attempt = now
        payload = {
            "camera_id": camera_id,
            "facility_id": camera.facility_id,
            "config_version": worker.version,
        }
        headers = {
            "Content-Type": "application/json",
            "X-Edge-Relay-Token": worker.relay.token.get_secret_value(),
        }
        try:
            result = bounded_request(
                worker.relay_heartbeat_url,
                "POST",
                headers,
                encode_json(payload),
                HEARTBEAT_TIMEOUT_SEC,
            )
        except FatalAcceleratorError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._record_failure(
                DeliveryFailure(
                    DeliveryDisposition.RETRY,
                    "UNEXPECTED",
                    transport_error=f"{type(exc).__name__}: {exc}",
                )
            )
            return
        if isinstance(result, DeliveryFailure):
            self._record_failure(result)
            return
        status, headers_out, _body = result
        if not 200 <= status < 300:
            self._record_failure(classify_http_failure(status, headers_out))
            return
        self._record_success()

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def _record_failure(self, failure: DeliveryFailure) -> None:
        self.failure_count += 1
        self._failure_log.record_failure(failure, path=RELAY_HEARTBEAT_PATH)

    def _record_success(self) -> None:
        self._failure_log.record_success(path=RELAY_HEARTBEAT_PATH)


@dataclass(frozen=True, slots=True)
@final
class _WindowGatedDecider:
    decider: Decider
    window: DetectionWindow
    clock: Callable[[], datetime]
    last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = ()
    last_update_evaluated: bool = True
    last_shadow_trace_count: int = 0

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        if not self.window.contains(self.clock()):
            object.__setattr__(
                self,
                "last_trace_snapshots",
                (
                    DecisionTraceSnapshot(
                        reason="outside-detection-window",
                        previous_state="not-evaluated",
                        current_state="not-evaluated",
                        triggered=False,
                        track_id=None,
                        bed_id=None,
                        missing_values={"decision_state": "outside-detection-window"},
                    ),
                ),
            )
            object.__setattr__(self, "last_update_evaluated", True)
            object.__setattr__(self, "last_shadow_trace_count", 0)
            return ()
        events = self.decider.update(input_value)
        if isinstance(self.decider, TraceSnapshotProvider):
            object.__setattr__(self, "last_trace_snapshots", self.decider.last_trace_snapshots)
            object.__setattr__(
                self,
                "last_update_evaluated",
                (
                    self.decider.last_update_evaluated
                    if isinstance(self.decider, FreshnessProvider)
                    else True
                ),
            )
            object.__setattr__(
                self,
                "last_shadow_trace_count",
                (
                    self.decider.last_shadow_trace_count
                    if isinstance(self.decider, ShadowTraceProvider)
                    else 0
                ),
            )
        return events


def _decision_identity_for(
    config: WorkerConfig, module_qualified_id: str
) -> DecisionIdentity | None:
    module_id = module_qualified_id.split(".v", 1)[0]
    policy = config.detection_policies.defaults.get(module_id)
    if policy is None:
        return None
    return DecisionIdentity(
        module_qualified_id=module_qualified_id,
        effective_policy_id=str(policy.effective_policy_id),
    )


def _absorbed_track_id_switch_total(decision: EventAggregator) -> int:
    for decider in decision.deciders:
        fall_decider = decider.decider if isinstance(decider, _WindowGatedDecider) else decider
        if isinstance(fall_decider, FallDomainDecider):
            return fall_decider.track_id_switch_absorbed_total
    raise RuntimeError("native policy decision lacks a fall absorbed-switch counter")


def _delivery_queue_dir(state_dir: Path) -> Path:
    return state_dir / "delivery-queue"


class ClipAnalysisDisabled:
    def status(self, clip_id: str) -> object:
        del clip_id
        raise ClipAnalysisDisabledError("clip_analysis_disabled")

    def trigger(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise ClipAnalysisDisabledError("clip_analysis_disabled")

    def enqueue(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return False

    def notify(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def cancel(self, clip_id: str) -> bool:
        del clip_id
        raise ClipAnalysisDisabledError("clip_analysis_disabled")

    def shutdown(self) -> None:
        return None


def _clip_analysis_cpu_index(environ: Mapping[str, str]) -> int | None:
    raw_value = environ.get(CLIP_ANALYSIS_CPU_ENV)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        cpu_index = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{CLIP_ANALYSIS_CPU_ENV} must be an integer") from exc
    if cpu_index < 0:
        raise RuntimeError(f"{CLIP_ANALYSIS_CPU_ENV} must be non-negative")
    return cpu_index


def _production_cuda_source() -> CudaProbe:
    capability = probe_cuda_capability()
    return CudaProbe(
        available=capability.available,
        reason=capability.reason,
        device_count=capability.device_count,
        arch_list=capability.arch_list,
    )


def _production_mps_source() -> bool:
    return probe_mps_capability().available


def production_boot_dependencies() -> bootstrap.BootDependencies:
    return bootstrap.BootDependencies(
        default_verifiers(
            cuda_source=_production_cuda_source,
            mps_source=_production_mps_source,
            device_resident_source=_production_device_resident_source,
        )
    )


def _production_device_resident_source() -> VerifyResult:
    status = probe_nvml_gpu_status()
    if not status.nvml_available:
        return VerifyResult(False, "flow", "device", status.reason)
    device = status.device_name or "an unnamed device"
    driver = status.driver_version or "an unreported driver"
    return VerifyResult(True, "flow", "device", f"NVML reports {device} on driver {driver}")


def _production_gpu_status(*, probe_python_cuda: bool = True) -> RelayGpuPayload:
    nvml_status = probe_nvml_gpu_status()
    cuda_context_ok = probe_cuda_capability().available if probe_python_cuda else False
    return RelayGpuPayload(
        nvml_available=nvml_status.nvml_available,
        cuda_context_ok=cuda_context_ok,
        driver_version=nvml_status.driver_version,
        device_name=nvml_status.device_name,
        captured_at_sec=time.time(),
        nvml_error=None if nvml_status.nvml_available else nvml_status.reason,
    )


@final
class NativeHeartbeatLoop:
    def __init__(
        self,
        worker: WorkerConfig,
        cameras: Sequence[CameraRuntimeConfig],
        pumps: Sequence[NativePolicyPump],
        *,
        tick_sec: float = 5.0,
    ) -> None:
        by_id = {camera.camera_id: camera for camera in cameras}
        self._pumps = tuple(pump for pump in pumps if pump.camera_id in by_id)
        self._reporters = {
            pump.camera_id: HeartbeatReporter(worker, by_id[pump.camera_id]) for pump in self._pumps
        }
        self._tick_sec = tick_sec
        self._seen: dict[str, int] = {pump.camera_id: pump.processed_count for pump in self._pumps}
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._tick_sec):
            for pump in self._pumps:
                camera_id = pump.camera_id
                processed = pump.processed_count
                advanced = processed > self._seen[camera_id]
                self._seen[camera_id] = processed
                if not advanced:
                    continue
                try:
                    self._reporters[camera_id].mark_ready(camera_id)
                except FatalAcceleratorError:
                    raise
                except Exception:
                    LOGGER.warning(
                        "native heartbeat failed: camera_id=%s", camera_id, exc_info=True
                    )

    def stop(self) -> None:
        self._stop.set()


@final
class WorkerRuntime:
    _flow_media_plane: FlowMediaPlane | None = None
    _flow_lifecycle_supervisor: FlowLifecycleSupervisor | None = None

    def __init__(
        self,
        config: WorkerConfig,
        *,
        serving_client: ServingClient,
        env: Mapping[str, str] | None = None,
        acquire_lease: bootstrap.LeaseAcquirer | None = None,
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,
        restart_check: Callable[[], bool] | None = None,
        clip_export_policy: LiveClipExportPolicy | None = None,
        max_frames_per_camera: int | None = None,
        state_dir: Path | None = None,
        clip_store_dir: Path | None = None,
        module_registry: CompiledDetectionModuleRegistry | None = None,
        restart_generation: int = 0,
        build_revision: str | None = None,
        environment_facts_factory: Callable[
            [BootContext, str | None], RuntimeEnvironmentFacts
        ] = collect_runtime_environment_facts,
        temporal_profile: TemporalProfile = CURRENT_TEMPORAL_PROFILE,
        flow_media_plane: FlowMediaPlane | None = None,
    ) -> None:
        self.config = config
        self.temporal_profile = temporal_profile
        self._module_registry = module_registry or DETECTION_MODULE_REGISTRY
        self._module_versions = config.domains.selected_versions(self._module_registry)
        self._restart_generation = restart_generation
        self._build_revision = build_revision
        self._environment_facts_factory = environment_facts_factory
        self._worker_boot_uuid = uuid.uuid4()
        self._boot_instance_id = f"worker:{self._worker_boot_uuid}"
        self._runtime_manifest: AppliedRuntimeManifest | None = None
        self._clip_store_dir = (
            Path(DEFAULT_CLIP_STORE_DIR) if clip_store_dir is None else clip_store_dir
        )
        resolved_domain_names = tuple(self._module_versions)
        domain_source = (
            "config override" if self.config.domains.resolved_overrides() else "registry default"
        )
        LOGGER.info(
            "resolved active detection domains (%s): %s",
            domain_source,
            ", ".join(resolved_domain_names) or "(none)",
        )
        if not resolved_domain_names:
            LOGGER.warning("resolved active detection domains is empty; no detection will run")
        self._env = os.environ if env is None else env
        self._serving = serving_client
        self._state_dir = state_dir if state_dir is not None else resolve_state_dir()
        LOGGER.info("worker state directory resolved to %s", self._state_dir)
        self._acquire = acquire_lease or (lambda: GpuLease.acquire(self._state_dir))
        self._boot_dependencies = boot_dependencies or production_boot_dependencies()
        self._hard_exit = hard_exit
        self._restart_check = restart_check
        self._clip_export_policy = clip_export_policy or LiveClipExportPolicy(
            config.clip_export_enabled,
            config.clip_export_version,
        )
        self._max_frames_per_camera = max_frames_per_camera
        self._context = bootstrap.BootstrapContext()
        self._boot: BootContext | None = None
        self.fall_model: FallModelProtocol | None = None
        self._loaded_fall_bundle: ort_pose_bbox56.PackagedFallBundle | None = None
        self._shared_graph: SharedComponentGraph | None = None
        self._warmed_component_ids: frozenset[str] = frozenset()
        self.fault_handler: FaultHandler | None = None
        self.watchdog: InferenceWatchdog | None = None
        self._evidence_export_runtime: EvidenceExportRuntime | None = None
        self._runtime_status_sender: RuntimeStatusSender | None = None
        self.diagnostics = WorkerDiagnostics()
        self.diagnostics.set_gpu_status(_production_gpu_status(probe_python_cuda=False))
        self._snapshot_store = SnapshotStore(self._resolved_clip_store_dir())
        self._camera_evidence_attachers: dict[str, AlertEvidenceAttacher] = {}
        self._mjpeg_config = self._resolve_mjpeg_config()
        self._live_frames = LatestFrameStore()
        self._mjpeg_server: MjpegServer | None = None
        self._clip_analysis_supervisor: ClipAnalysisControl | None = None
        self._clip_analysis_catchup_thread: threading.Thread | None = None
        self._clip_analysis_catchup_stop = threading.Event()
        self._flow_media_plane: FlowMediaPlane | None = flow_media_plane
        self._flow_lifecycle_supervisor: FlowLifecycleSupervisor | None = None
        self._native_policy_pumps: tuple[NativePolicyPump, ...] = ()
        self._native_policy_pumps_by_camera: dict[str, NativePolicyPump] = {}
        self._policy_pump_threads: tuple[threading.Thread, ...] = ()
        self._selected_bundle_admission: ModelBundleProof | None = None
        self._execution_record_lanes = None
        self._execution_record_exporter = None

    def _stop_flow_media_plane(self) -> None:
        if self._flow_media_plane is None:
            return
        self._live_frames.set_demand_listener(None)
        try:
            self._flow_media_plane.stop()
        except DeepStreamFlowStopTimeout:
            LOGGER.critical("Flow shutdown deadline exceeded; terminating worker for restart")
            self._hard_exit(FATAL_ACCELERATOR_EXIT_CODE)
            raise
        self._flow_media_plane = None

    def _resolved_clip_store_dir(self) -> Path:
        base = self._clip_store_dir
        subdir = self.config.clip.store_subdir
        if not subdir:
            return base
        candidate = PurePosixPath(subdir)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError("clip store subdirectory must be relative and traversal-free")
        return base / subdir

    def _resolve_mjpeg_config(self) -> MjpegServerConfig:
        configured = self.config.dev_mjpeg
        source = (
            MjpegServerConfig(enabled=True, host=configured.host, port=configured.port)
            if configured.enabled
            else dev_mjpeg_config(self._env)
        )
        return MjpegServerConfig(
            enabled=source.enabled,
            host=source.host,
            port=source.port,
            probe_token=self.config.relay.token.get_secret_value(),
        )

    def run(self) -> None:
        def decode_probe(_decode: str) -> VerifyResult:
            return VerifyResult(True, "flow", "decode", "Flow media plane")

        def encode_probe() -> VerifyResult:
            return VerifyResult(True, "flow", "encode", "Flow media plane")

        stages = bootstrap.named_stages(
            self._context,
            self._env,
            initializers={"models": self._initialize_models},
            warmups={"models": lambda _models: self._warm_models()},
            activate=self._activate,
            decode_probe=decode_probe,
            encode_probe=encode_probe,
            deps=self._boot_dependencies,
            acquire=self._acquire,
        )
        try:
            _ = bootstrap.bootstrap_or_exit(stages, context=self._context)
            self.diagnostics.set_worker_status(
                RelayWorkerPayload(
                    alive=True,
                    pid=os.getpid(),
                    started_at_sec=time.time(),
                    profile_boot_error=None,
                )
            )
            self._start_export_sender()
            if self._execution_record_exporter is not None:
                self._execution_record_exporter.start()
            self._start_runtime_status_sender()
            self._start_live_view_server()
            while not self._max_frames_completion_check() and (
                not self._restart_check or not self._restart_check()
            ):
                time.sleep(1)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_flow_media_plane()
        for pump in self._native_policy_pumps:
            pump.stop()
        for thread in self._policy_pump_threads:
            thread.join(timeout=5.0)
        self._policy_pump_threads = ()
        if self.watchdog is not None:
            self.watchdog.stop()
        if self._evidence_export_runtime is not None:
            self._evidence_export_runtime.stop_sender()
        if self._execution_record_exporter is not None:
            self._execution_record_exporter.stop()
        if self._runtime_status_sender is not None:
            self._runtime_status_sender.stop()
        if self._mjpeg_server is not None:
            self._mjpeg_server.stop()
            self._mjpeg_server = None
        if self._clip_analysis_catchup_thread is not None:
            self._clip_analysis_catchup_stop.set()
            self._clip_analysis_catchup_thread.join(timeout=5.0)
            if self._clip_analysis_catchup_thread.is_alive():
                LOGGER.error("clip analysis catch-up thread did not stop within timeout")
            else:
                self._clip_analysis_catchup_thread = None
        if self._clip_analysis_supervisor is not None:
            self._clip_analysis_supervisor.shutdown()
            self._clip_analysis_supervisor = None
        self._context.release_lease()

    def _start_live_view_server(self) -> None:
        if self._boot is None:
            raise RuntimeError("live view server cannot start before flow boot")
        if not self._mjpeg_config.enabled:
            LOGGER.info("dev_mjpeg disabled; live view server not started")
            return
        clip_store_dir = self._clip_store_dir
        cpu_index = _clip_analysis_cpu_index(self._env)
        supervisor = (
            ClipAnalysisDisabled()
            if cpu_index is None
            else ClipAnalysisSupervisor(
                python_executable=sys.executable,
                pose_model_path=Path(self._env["ML_WORKER_FLOW_ONNX_PATH"]),
                bed_model_path=Path(BED_ONNX_MODEL_PATH),
                profile=ClipAnalysisProfile(
                    person_threshold=0.25,
                    bed_confidence=BED_MODEL_CONFIDENCE,
                    max_frames=5400,
                    max_duration_s=180.0,
                    max_pixels=3840 * 2160,
                    max_input_bytes=128 * 1024 * 1024,
                ),
                cpu_index=cpu_index,
                deadline_s=600.0,
            )
        )
        self._mjpeg_server = start_optional_mjpeg_server(
            self._live_frames,
            self._mjpeg_config,
            clip_analysis_supervisor=supervisor,
            clip_store_dir=clip_store_dir,
            bed_zone_recognizer=NvidiaBedZoneRecognizer(
                self._serving,
                timeout_s=DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S,
            ),
            bed_zone_snapshot=(
                self._flow_media_plane.native_snapshot
                if self._flow_media_plane is not None
                else None
            ),
            replay_fall_model=self.fall_model,
        )
        if self._mjpeg_server is None:
            supervisor.shutdown()
            LOGGER.warning(
                "live view enabled but its server could not bind: host=%s port=%d",
                self._mjpeg_config.host,
                self._mjpeg_config.port,
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )
        else:
            self._clip_analysis_supervisor = supervisor
            if not isinstance(supervisor, ClipAnalysisDisabled):
                self._clip_analysis_catchup_stop.clear()
                self._clip_analysis_catchup_thread = start_clip_analysis_catchup(
                    clip_store_dir, supervisor, self._clip_analysis_catchup_stop
                )
            LOGGER.info(
                "live view server bound: host=%s port=%d",
                self._mjpeg_config.host,
                self._mjpeg_server.port,
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )

    def _max_frames_completion_check(self) -> bool:
        cap = self._max_frames_per_camera
        return (
            cap is not None
            and bool(self._native_policy_pumps)
            and all(pump.processed_count >= cap for pump in self._native_policy_pumps)
        )

    def _start_runtime_status_sender(self) -> None:
        facility_by_camera: Mapping[str, str] = MappingProxyType(
            {camera.camera_id: camera.facility_id for camera in self.config.cameras}
        )
        transport = RelayRuntimeStatusTransport(
            self.config.relay.url,
            self.config.relay.token.get_secret_value(),
            request=runtime_status_sender_module.bounded_request,
        )
        sender = RuntimeStatusSender(
            self.diagnostics,
            facility_by_camera,
            transport,
            before_publish=self._refresh_runtime_status_telemetry,
            delivery_queue=DeliveryQueue(_delivery_queue_dir(self._state_dir)),
        )
        try:
            sender.start()
        except Exception:
            LOGGER.warning("runtime status sender failed to start", exc_info=True)
            return
        self._runtime_status_sender = sender

    def _initialize_models(self, boot: BootContext) -> SharedComponentGraph:
        self._boot = boot
        self._admit_selected_fall_bundle()
        self.fault_handler = FaultHandler(
            boot.profile.name, hard_exit=self._hard_exit, state_dir=self._state_dir
        )
        self.watchdog = InferenceWatchdog(self.fault_handler, profile=boot.profile.name)
        return self._initialize_flow_media_plane(boot)

    def _fall_models(self) -> WorkerModelsConfig:
        models = self.config.models
        if models is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        return models

    def _admit_selected_fall_bundle(self) -> None:
        models = self._fall_models()
        selected = models.selected
        if selected is None:
            return
        if models.box_source != "pose":
            raise RuntimeError("selected fall bundle requires box_source=pose")
        self._selected_bundle_admission = admit_model_bundle(selected.models_root, selected.desired)

    def _initialize_flow_media_plane(self, boot: BootContext) -> SharedComponentGraph:
        self._flow_engine_identity = verify_flow_boot_inputs(
            self._env, deployed_batch=len(self.config.cameras)
        )
        if self._flow_media_plane is None:
            self._flow_media_plane = FlowMediaPlane(
                FlowMediaPlaneConfig(
                    infer_config_path=self._env["ML_WORKER_FLOW_INFER_CONFIG"],
                    tracker_config_path=self._env["ML_WORKER_FLOW_TRACKER_CONFIG"],
                    tracker_library_path=self._env["ML_WORKER_FLOW_TRACKER_LIBRARY"],
                    record_dir=Path(self._env["ML_WORKER_FLOW_RECORD_DIR"]),
                    record_cache_seconds=int(self._env["ML_WORKER_FLOW_RECORD_CACHE_SECONDS"]),
                    frame_width=int(self._env["ML_WORKER_FLOW_FRAME_WIDTH"]),
                    frame_height=int(self._env["ML_WORKER_FLOW_FRAME_HEIGHT"]),
                    snapshot_branch_enabled=True,
                    source_silence_timeout_sec=float(
                        self._env.get(
                            "ML_WORKER_FLOW_SOURCE_SILENCE_TIMEOUT_SEC",
                            str(FlowLifecycleSupervisor.DEFAULT_SILENCE_TIMEOUT_SEC),
                        )
                    ),
                ),
                bed_zone_geometry={
                    camera.camera_id: BedZoneGeometry(
                        polygons=tuple(region.polygon for region in camera.bed_zone_regions),
                        image_width=camera.bed_zone_image_width,
                        image_height=camera.bed_zone_image_height,
                    )
                    for camera in self.config.cameras
                    if camera.bed_zone_regions
                    and camera.bed_zone_image_width is not None
                    and camera.bed_zone_image_height is not None
                },
                renderer=PreviewRenderer(),
                fall_states=self._fall_preview_states,
                worker_boot_id=str(self._worker_boot_uuid),
            )
        self._flow_media_plane.bind_live_frames(self._live_frames)
        return self._initialize_flow_policy_graph(boot)

    def _initialize_flow_policy_graph(self, boot: BootContext) -> SharedComponentGraph:
        fall_model = self._create_fall_model()
        models = self._fall_models()
        flags = {"person-box-source": models.box_source == "person"}
        bindings = self._module_registry.shared_bindings(self._module_versions, flags=flags)
        components: dict[str, object] = {}
        identities: list[SharedComponentIdentity] = []
        for binding in bindings:
            if binding.component_id == "fall-classifier":
                bundle = self._loaded_fall_bundle
                if bundle is None:
                    raise RuntimeError("flow policy requires a loaded fall bundle")
                digest = bundle.published_weights_digest
                preprocessing = bundle.preprocessing_identity
                components[binding.component_id] = fall_model
                identities.append(
                    SharedComponentIdentity(
                        binding.component_id, digest, "cpu-policy", "cpu", preprocessing
                    )
                )
                continue
            digest = binding.artifact_digest
            preprocessing = binding.preprocessing_identity
            if not isinstance(digest, str) or not digest or not preprocessing:
                raise RuntimeError(f"flow component {binding.component_id!r} has no identity")
            components[binding.component_id] = _NativeEngineComponent(digest, preprocessing)
            identities.append(
                SharedComponentIdentity(
                    binding.component_id,
                    digest,
                    "deepstream-flow" if binding.component_id == "pose" else "onnxruntime-cpu",
                    boot.device if binding.component_id == "pose" else "cpu",
                    preprocessing,
                )
            )
        graph = SharedComponentGraph(MappingProxyType(components), (), tuple(identities), None)
        self._shared_graph = graph
        self.fall_model = fall_model
        self._warmed_component_ids = frozenset(graph.components)
        self._compose_execution_records()
        return graph

    def _packaged_fall_member_digest(self) -> str:
        models = self._fall_models()
        configured = models.fall
        if configured is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        bundle = self._loaded_fall_bundle
        if bundle is None:
            bundle = ort_pose_bbox56.load_packaged_fall_bundle(configured.artifact_dir)
            _validate_fall_bundle_conformance(bundle.runner.conformance)
            self._loaded_fall_bundle = bundle
        return bundle.published_weights_digest

    def _create_fall_model(self) -> FallModelProtocol:
        models = self._fall_models()
        selected = models.selected
        if selected is not None:
            selection = selected.desired.selection
            if selection is None:
                raise RuntimeError("selected fall bundle has no selection contract")
            artifact_dir = selected.models_root / "bundles" / selected.desired.bundle_sha256
            if selection.runtime_format != "onnxruntime":
                raise RuntimeError(
                    "flow profile refuses a Torch fall bundle; the selected runtime_format "
                    "must be onnxruntime (export model.onnx with worker.tools.export_fall_onnx)"
                )
            proof = self._selected_bundle_admission
            if proof is None:
                raise RuntimeError("selected fall bundle must be admitted before construction")
            runner = ort_pose_bbox56.OrtPoseBbox56Runner.from_admitted_bundle(
                artifact_dir, proof, selection
            )
            _validate_fall_bundle_conformance(runner.conformance)
            self._loaded_fall_bundle = ort_pose_bbox56.PackagedFallBundle(
                runner, runner.artifact_digest, runner.preprocessing_identity
            )
            return runner

        configured = models.fall
        if configured is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        if configured.framework != "onnxruntime":
            raise RuntimeError(
                "flow profile requires the ONNX Runtime fall model; the packaged config "
                f"resolved framework={configured.framework!r} (export model.onnx with "
                "worker.tools.export_fall_onnx and boot with ML_WORKER_PROFILE=flow)"
            )
        bundle = ort_pose_bbox56.load_packaged_fall_bundle(configured.artifact_dir)
        _validate_fall_bundle_conformance(bundle.runner.conformance)
        self._loaded_fall_bundle = bundle
        return bundle.runner

    def _warm_models(self) -> tuple[str, ...]:
        if self._shared_graph is None or self._boot is None:
            raise RuntimeError("models cannot warm before initialization")
        if self.fall_model is None or self._flow_media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        self._warm_one(self.fall_model, "cpu")
        return tuple(sorted(self._warmed_component_ids))

    def _warm_one(self, model: RunnerProtocol | FallModelProtocol, device: str) -> None:
        if not isinstance(model, _Warmable):
            raise TypeError("configured model does not expose warmup")
        _ = warmup_to_ready(model, device=device)

    def _activate(self, boot: BootContext) -> tuple[bootstrap.CameraStageOutcome, ...]:
        handler = self.fault_handler
        if handler is None:
            raise RuntimeError("camera activation requires initialized fault handler")
        return self._activate_flow(boot, handler)

    def _activate_flow(
        self,
        boot: BootContext,
        handler: FaultHandler,
    ) -> tuple[bootstrap.CameraStageOutcome, ...]:
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        self._compose_evidence_export()
        plans = {
            camera.camera_id: self._preflight_camera_graph(camera) for camera in self.config.cameras
        }
        self._apply_runtime_manifest(boot, plans)
        pumps: list[NativePolicyPump] = []
        self._native_policy_pumps_by_camera.clear()
        sealed_bindings: list[FlowEvidenceBinding] = []
        outcomes = tuple(
            bootstrap.run_camera_stage(
                camera.camera_id,
                partial(self._build_flow_camera, camera, pumps, sealed_bindings),
            )
            for camera in self.config.cameras
        )
        self._replay_sealed_clips(sealed_bindings)
        media_plane.start()
        self._await_flow_first_frame(pumps)
        self._native_policy_pumps = tuple(pumps)
        for pump in pumps:
            handler.register_loop(pump)
        cameras = {camera.camera_id: camera for camera in self.config.cameras}
        reporters = {
            camera_id: HeartbeatReporter(self.config, camera)
            for camera_id, camera in cameras.items()
        }

        def on_fatal(error: str) -> None:
            LOGGER.error("flow media plane fatal: error=%s", error)
            exc = FatalAcceleratorError(error, task="flow_media_plane")
            handler.handle(
                exc,
                make_fault_record(
                    exc,
                    profile="flow",
                    task="flow_media_plane",
                    stage="flow_media_plane",
                ),
            )

        def on_unready(camera_id: str) -> None:
            LOGGER.warning("flow source outage: camera_id=%s category=metadata_silence", camera_id)

        self._flow_lifecycle_supervisor = FlowLifecycleSupervisor(
            media_plane,
            cameras,
            on_ready=lambda camera_id: reporters[camera_id].mark_ready(camera_id),
            on_unready=on_unready,
            on_fatal=on_fatal,
            silence_timeout_sec=media_plane.config.source_silence_timeout_sec,
        )
        heartbeat = NativeHeartbeatLoop(self.config, self.config.cameras, pumps)
        handler.register_loop(heartbeat)
        threading.Thread(target=heartbeat.run, name="flow-heartbeat", daemon=True).start()
        self._policy_pump_threads = tuple(
            threading.Thread(
                target=pump.run,
                name=f"flow-policy-{pump.camera_id}",
                daemon=True,
            )
            for pump in pumps
        )
        for thread in self._policy_pump_threads:
            thread.start()
        return outcomes

    def _compose_evidence_export(self) -> None:
        probe_camera_id = self.config.cameras[0].camera_id if self.config.cameras else "worker"
        try:
            runtime = EvidenceExportRuntime.from_config(
                store_dir=self._resolved_clip_store_dir(),
                queue_directory=_delivery_queue_dir(self._state_dir),
                relay_url=self.config.relay.url,
                relay_token=self.config.relay.token.get_secret_value(),
                probe_camera_id=probe_camera_id,
                clip_export_enabled=self._clip_export_policy.enabled,
                flow_sealed_sidecar_directory=self._state_dir / "flow-sealed",
                execution_records=self._execution_record_lanes,
                observing_boot_id=(
                    None if self._execution_record_lanes is None else str(self._worker_boot_uuid)
                ),
            )
            with ClipStoreLock.acquire(self._resolved_clip_store_dir()):
                runtime.initialize_under_lock()
        except ClipStoreLockedError as exc:
            raise EvidenceDeliveryError(
                "clip store is locked by another process; refusing evidence delivery"
            ) from exc
        except ValueError as exc:
            raise EvidenceDeliveryError(
                "evidence delivery is misconfigured: relay URL, relay token, "
                "and a probe identity are required"
            ) from exc
        except Exception as exc:
            raise EvidenceDeliveryError(
                "evidence delivery failed to initialize under the clip-store lock"
            ) from exc
        self._evidence_export_runtime = runtime

    def _start_export_sender(self) -> None:
        if self._evidence_export_runtime is None:
            raise EvidenceDeliveryError("evidence delivery was not composed")
        try:
            self._evidence_export_runtime.start_sender()
        except Exception as exc:
            raise EvidenceDeliveryError("evidence export sender failed to start") from exc

    def _compose_execution_records(self) -> None:
        from worker.runtime.config.execution_records import (
            execution_records_settings_from_environment,
        )

        if execution_records_settings_from_environment(self._env) is None:
            self._execution_record_lanes = None
            self._execution_record_exporter = None
            return
        bundle = self._loaded_fall_bundle
        identity = getattr(self, "_flow_engine_identity", None)
        image_digest = identity.get("image_digest") if isinstance(identity, Mapping) else None
        policy = self.config.detection_policies.defaults.get("fall")
        _settings, lanes, exporter = compose_execution_records(
            self.config,
            env=self._env,
            build_revision=self._build_revision,
            image_digest=image_digest if isinstance(image_digest, str) else None,
            model_digest=None if bundle is None else bundle.published_weights_digest,
            calibration_digest=self._fall_calibration_digest(),
            preprocessing_identity=None if bundle is None else bundle.preprocessing_identity,
            policy_identity=None if policy is None else policy.effective_policy_id,
        )
        self._execution_record_lanes = lanes
        self._execution_record_exporter = exporter
        plane = self._flow_media_plane
        if plane is not None:
            plane.metadata.set_execution_record_sink(lanes)

    def _fall_calibration_digest(self) -> str | None:
        models = self._fall_models()
        root = None if models.fall is None else models.fall.artifact_dir
        if root is None and models.selected is not None:
            root = models.selected.models_root / "bundles" / models.selected.desired.bundle_sha256
        if root is None:
            return None
        try:
            from worker.adapters.model.pose_bbox56_bundle_support import member_digest, read_json

            return member_digest(read_json(root / "bundle-manifest.json"), "calibration.json")
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _replay_sealed_clips(bindings: Sequence[FlowEvidenceBinding]) -> int:
        failures = 0
        for binding in bindings:
            try:
                binding.replay_sealed()
            except Exception:
                failures += 1
                LOGGER.exception(
                    "replaying a sealed clip failed for camera_id=%s; the media and its "
                    "sidecar are retained and cameras continue to activate",
                    binding.camera_id,
                )
        return failures

    def _await_flow_first_frame(
        self, pumps: list[NativePolicyPump], *, timeout_sec: float = 30.0
    ) -> None:
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        cameras = {camera.camera_id: camera for camera in self.config.cameras}
        pending = [
            (pump.camera_id, binding)
            for pump, binding in (
                (pump, media_plane.metadata.expected_binding(pump.camera_id)) for pump in pumps
            )
            if binding is not None
        ]
        if not pending:
            raise FlowWarmupTimeout("Flow warmup has no registered source to wait for")
        deadline = time.monotonic() + timeout_sec
        tokens = {camera_id: media_plane.metadata.subscribe(b) for camera_id, b in pending}
        ready: set[str] = set()
        while time.monotonic() < deadline and len(ready) < len(tokens):
            for camera_id, token in tokens.items():
                if camera_id in ready:
                    continue
                try:
                    _ = media_plane.metadata.wait_accepted(token, timeout_sec=1.0)
                except TimeoutError:
                    continue
                ready.add(camera_id)
                HeartbeatReporter(self.config, cameras[camera_id]).mark_ready(camera_id)
        if not ready:
            raise FlowWarmupTimeout(
                "Flow warmup did not receive an accepted metadata frame from any source"
            )
        unproven = sorted(set(tokens) - ready)
        if unproven:
            LOGGER.warning(
                "flow warmup: %d of %d cameras published a frame; still waiting on %s",
                len(ready),
                len(tokens),
                ", ".join(unproven),
            )

    def _on_clip_ready(self, publication: ReadyClipPublication) -> None:
        supervisor = self._clip_analysis_supervisor
        if supervisor is None:
            return
        try:
            supervisor.notify(
                publication.clip_id,
                publication.video_path,
                publication.sha256,
                size_bytes=publication.size_bytes,
                duration_ms=publication.duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "clip analysis ready hook failed stage=clip_analysis_ready clip_id=%s "
                "exception_class=%s",
                publication.clip_id,
                type(exc).__name__,
            )

    def _build_flow_camera(
        self,
        camera: CameraRuntimeConfig,
        pumps: list[NativePolicyPump],
        sealed_bindings: list[FlowEvidenceBinding],
    ) -> None:
        from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

        if camera.decode_backend not in {None, "auto", "nvdec"}:
            raise RuntimeError("flow cameras cannot override decode to a host backend")
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        endpoint = assert_rtsp_endpoint_allowed(camera.inference_rtsp_url)
        self.diagnostics.register_decode(camera.camera_id, camera.decode_backend or "auto")
        self._live_frames.register_camera(camera.camera_id)
        binding = media_plane.add_source(camera.camera_id, endpoint.pinned_url)
        plan = self._preflight_camera_graph(
            camera,
            episode_source_identity=(
                str(self._worker_boot_uuid),
                str(binding.stream_epoch),
                binding.source_generation,
            ),
        )
        scene = SceneState(
            camera.camera_id,
            persisted_bed_regions=_persisted_bed_regions(camera),
            bed_zone_image_width=camera.bed_zone_image_width,
            bed_zone_image_height=camera.bed_zone_image_height,
        )
        attacher = AlertEvidenceAttacher(
            domain_audit=plan.domain_audit,
            snapshot_renderer=None,
            debug_snapshots_provider=_debug_snapshots_provider(
                plan.domain_deciders, plan.definitions
            ),
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        self._camera_evidence_attachers[camera.camera_id] = attacher
        stager = DurableEvidenceStager(
            queue_directory=_delivery_queue_dir(self._state_dir),
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            resident_id=camera.resident_id,
            config_version=self.config.version,
            clock=time.time,
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        sealed_binding: list[FlowEvidenceBinding] = []
        actor = media_plane.smart_recorder(
            camera.camera_id,
            sink=lambda sealed: sealed_binding[0].on_sealed(sealed),
        )
        sink = FlowEvidenceBinding(
            actor=actor,
            stager=stager,
            publisher=FlowClipPublisher(
                ClipIdAllocator(self._resolved_clip_store_dir()),
                ClipPublisher(
                    self._resolved_clip_store_dir(),
                    delivery_queue_directory=_delivery_queue_dir(self._state_dir),
                    thumbnail_generator=FfmpegThumbnailGenerator(),
                    on_ready=self._on_clip_ready,
                ),
            ),
            sidecars=FlowSealedSidecars(self._state_dir / "flow-sealed"),
            camera_id=camera.camera_id,
            execution_records=self._execution_record_lanes,
        )
        sealed_binding.append(sink)
        sealed_bindings.append(sink)
        pump = NativePolicyPump(
            binding,
            NativePolicyContext(
                media_plane.metadata,
                media_plane,
                scene,
                plan.decision,
                sink,
                attacher,
                self.diagnostics,
                plan.schedule.get("bed", self.temporal_profile.decision_interval_frames("bed")),
                replay_trace=(
                    None
                    if (trace_directory := replay_trace_directory_from_environment()) is None
                    else ReplayTraceWriter(trace_directory, camera.camera_id)
                ),
                night_window_active=_night_window_active(plan.detection_windows.get("bed_exit")),
                recreate_decision=lambda rebuilt: (
                    self._preflight_camera_graph(
                        camera,
                        episode_source_identity=(
                            str(self._worker_boot_uuid),
                            str(rebuilt.stream_epoch),
                            rebuilt.source_generation,
                        ),
                        incidents=plan.decision.incidents,
                    ).decision
                ),
                track_id_switch_absorbed_total=_absorbed_track_id_switch_total,
                execution_records=self._execution_record_lanes,
            ),
        )
        self._native_policy_pumps_by_camera[camera.camera_id] = pump
        pumps.append(pump)
        self.diagnostics.register_native_detection(camera.camera_id)

    def _fall_preview_states(self, camera_id: str) -> Mapping[int, FallPreviewState]:
        pump = self._native_policy_pumps_by_camera.get(camera_id)
        return {} if pump is None else pump.preview_states()

    def _apply_runtime_manifest(
        self,
        boot: BootContext,
        plans: Mapping[str, CameraDetectionPlan],
    ) -> None:
        graph = self._shared_graph
        if graph is None:
            raise RuntimeError("runtime provenance requires initialized components")
        try:
            cameras = tuple(
                build_applied_camera_state(
                    camera_id=camera.camera_id,
                    effective_decode_backend=self._boot.runtime_profile.effective_decode_backend,
                    ingest_target_fps=self.temporal_profile.target_fps,
                    module_qualified_ids=tuple(
                        definition.qualified_id
                        for definition in plans[camera.camera_id].definitions.values()
                    ),
                    schedule=plans[camera.camera_id].schedule,
                    detection_windows={
                        module_id: (
                            None
                            if window is None
                            else AppliedDetectionWindow(window.start, window.end, window.tz)
                        )
                        for module_id, window in plans[camera.camera_id].detection_windows.items()
                    },
                    policies=MappingProxyType(
                        {
                            definition.module_id: self.config.detection_policies.resolve(
                                camera.camera_id,
                                definition.module_id,
                                definition.version,
                            )
                            for definition in plans[camera.camera_id].definitions.values()
                        }
                    ),
                    bed_zone_regions=camera.bed_zone_regions,
                    bed_zone_image_width=camera.bed_zone_image_width,
                    bed_zone_image_height=camera.bed_zone_image_height,
                )
                for camera in self.config.cameras
            )
            self._runtime_manifest = build_applied_runtime_manifest(
                boot=boot,
                module_registry=self._module_registry,
                module_versions=self._module_versions,
                component_identities=graph.identities,
                cameras=cameras,
                config_version=self.config.version,
                restart_generation=self._restart_generation,
                detector_version=DETECTOR_VERSION,
                environment=self._environment_facts_factory(boot, self._build_revision),
                edge_database_schema_version=EDGE_DATABASE_SCHEMA_VERSION,
            )
            AppliedRuntimeManifestStore(self._state_dir / "runtime-manifest").persist(
                self._runtime_manifest,
                boot_instance_id=self._boot_instance_id,
                applied_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            self._runtime_manifest = None
            if self._selected_bundle_admission is not None:
                raise
            LOGGER.warning(
                "runtime provenance could not be applied; continuing without it",
                exc_info=True,
            )

    def _refresh_runtime_status_telemetry(self) -> None:
        enabled, version = self._clip_export_policy.snapshot()
        self.diagnostics.set_clip_export_applied(enabled=enabled, version=version)
        self._refresh_flow_recording_telemetry()

    def _refresh_flow_recording_telemetry(self) -> None:
        media_plane = self._flow_media_plane
        if media_plane is None:
            return
        lifecycle = self._flow_lifecycle_supervisor
        if lifecycle is None:
            raise RuntimeError("flow lifecycle supervisor is not initialized")
        lifecycle.tick()
        status = media_plane.status()
        for source in status.sources:
            extended, extension_raced, start_refused = media_plane.recorder_counters(
                source.camera_id
            )
            self.diagnostics.record_flow_recording_counters(
                source.camera_id,
                extended=extended,
                extension_raced=extension_raced,
                start_refused=start_refused,
            )
            self.diagnostics.record_flow_nvenc_sessions(
                source.camera_id, status.nvenc_sessions_active
            )
            counters = lifecycle.counters(source.camera_id)
            self.diagnostics.record_flow_lifecycle_counters(
                source.camera_id,
                outages=counters.outages,
                recoveries=counters.recoveries,
            )
            LOGGER.info(
                "flow lifecycle: camera_id=%s outages=%d recoveries=%d",
                source.camera_id,
                counters.outages,
                counters.recoveries,
            )

    def _active_domain_names(self) -> tuple[str, ...]:
        return tuple(self._module_versions)

    def _preflight_camera_graph(
        self,
        camera: CameraRuntimeConfig,
        tracker: GreedyIouTracker | None = None,
        episode_source_identity: tuple[str, str, int] | None = None,
        incidents: IncidentManager | None = None,
    ) -> CameraDetectionPlan:
        graph = self._shared_graph
        if graph is None:
            raise RuntimeError("detection graph preflight requires initialized components")
        persisted_bed_regions = _persisted_bed_regions(camera)
        flags = {
            "person-box-source": self._fall_models().box_source == "person",
            "persisted-bed-region": bool(persisted_bed_regions),
        }
        activation = self._module_registry.activation(
            module_versions=self._module_versions,
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            available_component_ids=graph.components,
            warmed_component_ids=self._warmed_component_ids,
            output_adapter_ids=result_merger_names(),
            camera_frame_stride=camera.frame_stride,
            flags=flags,
            temporal_profile=self.temporal_profile,
        )
        if episode_source_identity is None:
            episode_source_identity = (str(self._worker_boot_uuid), "0", 0)
        camera_component_values: dict[str, object] = {
            "episode-identity": episode_source_identity,
        }
        if tracker is not None:
            camera_component_values["person-tracker"] = tracker
        camera_components: Mapping[str, object] = MappingProxyType(camera_component_values)
        domain_deciders: dict[str, Decider] = {}

        domain_identities: dict[str, DecisionIdentity | None] = {}
        domain_audit: dict[str, Mapping[str, object]] = {}
        definitions: dict[str, DetectionModuleDefinition] = {}
        detection_windows: dict[str, DetectionWindow | None] = {}
        for definition in activation.definitions:
            window = self._resolved_window(definition.module_id)
            detection_windows[definition.module_id] = window
            context = CameraModuleContext(
                camera_id=camera.camera_id,
                facility_id=camera.facility_id,
                shared_components=graph.components,
                camera_components=camera_components,
                detection_window=window,
                clock=lambda: datetime.now(UTC),
                diagnostics=self.diagnostics,
                policy=(
                    self.config.detection_policies.resolve(
                        camera.camera_id,
                        definition.module_id,
                        definition.version,
                    )
                    if definition.module_id in LATEST_POLICY_VERSIONS
                    else None
                ),
            )
            camera_module = definition.create_camera_module(context)
            camera_components = camera_module.camera_components
            decider = camera_module.decider
            if definition.window_mode == "external" and window is not None:
                decider = _WindowGatedDecider(decider, window, clock=lambda: datetime.now(UTC))
            domain_deciders[definition.module_id] = decider
            domain_identities[definition.module_id] = _decision_identity_for(
                self.config, definition.qualified_id
            )
            definitions[definition.module_id] = definition
            if definition.audit_adapter is not None:
                audit_context = replace(context, camera_components=camera_components)
                snapshot = definition.audit_adapter(audit_context)
                envelope = build_audit_envelope(
                    model_version=snapshot.model_version,
                    detector_version=DETECTOR_VERSION,
                    operating_threshold=snapshot.operating_threshold,
                )
                if snapshot.threshold_source is not None:
                    envelope["threshold_source"] = snapshot.threshold_source
                if snapshot.receipt_threshold is not None:
                    envelope["receipt_threshold"] = snapshot.receipt_threshold
                if snapshot.transition_votes is not None:
                    envelope["transition_votes"] = snapshot.transition_votes
                if snapshot.transition_window is not None:
                    envelope["transition_window"] = snapshot.transition_window
                if snapshot.confirmation_rule_source is not None:
                    envelope["confirmation_rule_source"] = snapshot.confirmation_rule_source
                if snapshot.receipt_transition_votes is not None:
                    envelope["receipt_transition_votes"] = snapshot.receipt_transition_votes
                if snapshot.receipt_transition_window is not None:
                    envelope["receipt_transition_window"] = snapshot.receipt_transition_window
                if snapshot.unapplied_transition_votes is not None:
                    envelope["unapplied_transition_votes"] = snapshot.unapplied_transition_votes
                if snapshot.unapplied_transition_window is not None:
                    envelope["unapplied_transition_window"] = snapshot.unapplied_transition_window
                if snapshot.unapplied_policy_threshold is not None:
                    self.diagnostics.record_fall_unapplied_policy_threshold(
                        camera.camera_id, snapshot.unapplied_policy_threshold
                    )
                domain_audit[definition.module_id] = envelope
        resolved_tracker = camera_components.get("person-tracker")
        if not isinstance(resolved_tracker, GreedyIouTracker):
            resolved_tracker = tracker or GreedyIouTracker()
        if incidents is None:
            incidents = IncidentManager(
                identity_path=event_identity_path(camera.camera_id, self._state_dir)
            )
        aggregator = EventAggregator(
            deciders=tuple(domain_deciders.values()),
            incidents=incidents,
            identities=tuple(domain_identities[name] for name in domain_deciders),
        )
        self.diagnostics.register_incident_manager(camera.camera_id, incidents)
        if _is_confirmed_cpu_fall_runner(self.fall_model):
            self.diagnostics.record_fall_inference_device(camera.camera_id, "cpu")
        return CameraDetectionPlan(
            tracker=resolved_tracker,
            schedule=activation.schedule,
            detection_windows=MappingProxyType(detection_windows),
            decision=aggregator,
            domain_audit=MappingProxyType(domain_audit),
            domain_deciders=MappingProxyType(domain_deciders),
            definitions=MappingProxyType(definitions),
        )

    def _build_decider(
        self,
        name: str,
        camera: CameraRuntimeConfig,
        fall_model: FallModelProtocol,
        tracker: GreedyIouTracker | None = None,
    ) -> Decider:
        definition = self._module_registry.get(name, self._module_versions.get(name))
        window = self._resolved_window(name)
        context = CameraModuleContext(
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            shared_components=MappingProxyType({"fall-classifier": fall_model}),
            camera_components=MappingProxyType(
                {
                    "person-tracker": tracker or GreedyIouTracker(),
                    "episode-identity": (str(self._worker_boot_uuid), "0", 0),
                }
            ),
            detection_window=window,
            clock=lambda: datetime.now(UTC),
            diagnostics=self.diagnostics,
            policy=self.config.detection_policies.resolve(
                camera.camera_id,
                definition.module_id,
                definition.version,
            ),
        )
        decider = definition.create_camera_module(context).decider
        if definition.window_mode == "external" and window is not None:
            return _WindowGatedDecider(decider, window, clock=lambda: datetime.now(UTC))
        return decider

    def _resolved_window(self, name: str) -> DetectionWindow | None:
        configured = self.config.domains.resolved_detection_window(name)
        if configured is None:
            return None
        return DetectionWindow(start=configured.start, end=configured.end, tz=configured.tz)


def _night_window_active(window: DetectionWindow | None) -> Callable[[], bool]:
    return (lambda: False) if window is None else lambda: window.contains(datetime.now(UTC))


def _is_confirmed_cpu_fall_runner(model: FallModelProtocol | None) -> bool:
    if model is None:
        return False
    device = getattr(model, "device", None)
    return str(device) == "cpu"


__all__ = [
    "HeartbeatReporter",
    "WorkerRuntime",
    "production_boot_dependencies",
]
