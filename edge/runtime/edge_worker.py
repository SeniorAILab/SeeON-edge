from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from contracts.event import EventPayload
from contracts.runner import RunnerProtocol
from contracts.worker_config import CONFIG_VERSION_KEY, PulledWorkerConfig
from edge.domains import DOMAIN_REGISTRY
from edge.domains.base import AuditContext, DomainDetector
from edge.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from edge.evidence.clip_store_lock import ClipStoreLockedError
from edge.evidence.event_identity import event_identity_path
from edge.evidence.evidence_runtime import EvidenceExportRuntime
from edge.perception.fall_window_classifier import FallModelProtocol
from edge.perception.tracker import GreedyIouTracker
from edge.runners.registry import DEFAULT_REGISTRY, ModelRegistry
from edge.runners.torch_lstm_fall import LstmFallRunner, ModelLoadError
from edge.runners.warmup import warmup_to_ready
from edge.runtime.camera_worker import CameraWorker, EvidenceStagerProtocol
from edge.runtime.config_pull import load_edge_worker_config_from_relay, pull_worker_config
from edge.runtime.config_resolver import apply_runtime_config, resolve_runtime_config
from edge.runtime.edge_worker_config import (
    EDGE_CAMERA_CONFIG_ENV,
    CameraRuntimeConfig,
    EdgeWorkerConfig,
    EdgeWorkerConfigError,
    load_edge_worker_config,
    resolve_config_path,
)
from edge.runtime.edge_worker_supervisor import EdgeWorkerSupervisor
from edge.runtime.mjpeg_server import (
    MjpegServer,
    OverlayFrameBuffer,
    OverlayPublisher,
    dev_mjpeg_enabled,
    dev_mjpeg_host,
    dev_mjpeg_port,
)
from edge.runtime.overlay_renderer import MAX_SNAPSHOT_BYTES, OverlayRenderer
from edge.runtime.pipeline_bootstrap import (
    GlobalBootstrapError,
    run_global_bootstrap,
)
from edge.runtime.profile import BootContext, profile_verify_stage
from edge.runtime.runtime_diagnostics import WorkerDiagnostics
from edge.runtime.runtime_status_sender import RuntimeStatusSender
from edge.runtime.scheduler import Scheduler
from edge.runtime.status_store import StatusStore
from edge.serving_client import InProcessServingClient, ServingClient
from edge.sources.rtsp import RTSPSource, create_backend
from edge.sources.rtsp_backend import RTSPBackend

# Phase-1 audit metadata identifies this worker-owned domain detector bundle.
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn(\.|$)")

DETECTOR_VERSION = "worker-domain-detectors-v1"
_PROFILE_BOOT_STATUS_TIMEOUT_SEC = 2.0


# Device + decode are resolved from ML_WORKER_PROFILE at boot
# (edge.runtime.profile); explicit, no silent CPU/decode fallback (ADR-0002).
@dataclass(frozen=True, slots=True)
class _Options:
    config_path: str | None
    check_config: bool
    max_frames_per_camera: int | None
    heartbeat_on_start: bool


@dataclass(frozen=True, slots=True)
class _StartupConfig:
    config: EdgeWorkerConfig
    registry_version: int
    source: str
    yaml_config: EdgeWorkerConfig | None = None
    pulled: PulledWorkerConfig | None = None


@dataclass(frozen=True, slots=True)
class _RunnerBundle:
    pose: RunnerProtocol
    bed: RunnerProtocol

    def as_mapping(self) -> Mapping[str, RunnerProtocol]:
        return {"pose": self.pose, "bed": self.bed}


@dataclass(frozen=True, slots=True)
class _WorkerResources:
    clients: Mapping[str, _RelayClient]
    runners: _RunnerBundle
    fall_model: FallModelProtocol
    status_store: StatusStore
    config: EdgeWorkerConfig
    overlay_publisher: OverlayPublisher | None = None
    stop_event: threading.Event | None = None
    clip_recorder: ClipRecorder | None = None
    diagnostics: WorkerDiagnostics | None = None
    evidence_stagers: Mapping[str, EvidenceStagerProtocol] = field(default_factory=dict)
    decode: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        options = _parse_args(args)
    except EdgeWorkerConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if options.check_config:
        try:
            startup = _load_startup_config(options)
        except EdgeWorkerConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        effective_config = startup.config
        print(
            json.dumps(
                {"ok": True, "cameras": len(effective_config.cameras)}, separators=(",", ":")
            )
        )
        return 0
    # Global profile-verify stage (ADR-0002): ML_WORKER_PROFILE (cuda|mps|cpu) is
    # explicit and owns both inference device and RTSP decode. A missing profile,
    # unusable device, or impossible device/decode combo fails fast to a non-zero
    # exit -- no silent CPU/OpenCV fallback.
    try:
        _boot = run_global_bootstrap([profile_verify_stage(os.environ)])
    except GlobalBootstrapError as exc:
        _publish_profile_boot_failure(os.environ, str(exc))
        print(
            f"Profile boot failed; refusing to start (no silent fallback): {exc}",
            file=sys.stderr,
        )
        return 3
    boot_context = _boot.outputs["profile_verify"]
    if not isinstance(boot_context, BootContext):  # pragma: no cover - defensive
        raise TypeError("profile_verify stage did not return a BootContext")
    startup_device = boot_context.device
    startup_decode = boot_context.decode
    try:
        startup = _load_startup_config(options)
    except EdgeWorkerConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    effective_config = startup.config
    config_version = startup.registry_version
    boot_registry_version = startup.registry_version
    for _camera in effective_config.cameras:
        _cam_decode = (_camera.decode_backend or "").strip().lower()
        if _cam_decode and _cam_decode != "auto" and _cam_decode != startup_decode:
            print(
                f"camera {_camera.camera_id} decode_backend={_cam_decode!r} conflicts "
                f"with profile decode {startup_decode!r}; refusing to start",
                file=sys.stderr,
            )
            return 3
    relay_url = os.environ.get("RELAY_URL", str(effective_config.relay.url))
    relay_token = os.environ.get("RELAY_TOKEN")
    print(f"worker config source: {startup.source}", file=sys.stderr)

    status_store = StatusStore()
    mjpeg_server: MjpegServer | None = None
    effective_relay_token = _effective_relay_token(effective_config, relay_token)
    if effective_config.dev_mjpeg.enabled or dev_mjpeg_enabled():
        overlay_buffer = OverlayFrameBuffer()
        for camera in effective_config.cameras:
            overlay_buffer.register_camera(camera.camera_id)
        mjpeg_server = MjpegServer(
            overlay_buffer,
            host=(
                effective_config.dev_mjpeg.host
                if effective_config.dev_mjpeg.enabled
                else dev_mjpeg_host()
            ),
            port=(
                effective_config.dev_mjpeg.port
                if effective_config.dev_mjpeg.enabled
                else dev_mjpeg_port()
            ),
            probe_token=effective_relay_token,
        )
        mjpeg_server.start()
        overlay_publisher = OverlayPublisher(overlay_buffer)
    else:
        overlay_publisher = None
    clip_config = ClipRecorderConfig()
    try:
        evidence_runtime = EvidenceExportRuntime.from_environment(
            store_dir=clip_config.store_dir,
            relay_url=relay_url,
            relay_token=effective_relay_token,
            probe_camera_id=effective_config.cameras[0].camera_id,
        )
    except (IndexError, ValueError) as exc:
        if mjpeg_server is not None:
            mjpeg_server.stop()
        print(
            f"evidence export configuration failed: {exc.__class__.__name__}",
            file=sys.stderr,
        )
        return 2
    clip_recorder = ClipRecorder(
        clip_config,
        is_clip_held=(None if evidence_runtime is None else evidence_runtime.is_clip_held),
        startup_hook=(None if evidence_runtime is None else evidence_runtime.initialize_under_lock),
        on_clip_finalized=(
            None if evidence_runtime is None else evidence_runtime.notify_clip_finalized
        ),
    )
    try:
        clip_recorder.start()
    except ClipStoreLockedError as exc:
        if mjpeg_server is not None:
            mjpeg_server.stop()
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - clip recording must not block worker startup
        if evidence_runtime is not None:
            if mjpeg_server is not None:
                mjpeg_server.stop()
            print(
                f"evidence export startup failed: {exc.__class__.__name__}",
                file=sys.stderr,
            )
            return 2
        print(f"clip recorder disabled: {exc}", file=sys.stderr)
        clip_recorder = None
    if evidence_runtime is not None:
        try:
            evidence_runtime.start_sender()
        except Exception as exc:  # noqa: BLE001 - enabled export startup fails closed
            if clip_recorder is not None:
                clip_recorder.stop()
            if mjpeg_server is not None:
                mjpeg_server.stop()
            print(
                f"evidence sender startup failed: {exc.__class__.__name__}",
                file=sys.stderr,
            )
            return 2
    evidence_stagers = (
        {}
        if evidence_runtime is None
        else {
            camera.camera_id: evidence_runtime.stager(
                camera_id=camera.camera_id,
                facility_id=camera.facility_id,
                resident_id=camera.resident_id,
                config_version=config_version,
            )
            for camera in effective_config.cameras
        }
    )
    diagnostics = WorkerDiagnostics(
        None if clip_recorder is None else clip_recorder.stats,
        gpu=_gpu_diagnostics(),
        worker={"alive": True, "pid": os.getpid(), "started_at_sec": time.time()},
    )
    for camera in effective_config.cameras:
        diagnostics.register_decode(camera.camera_id, startup_decode)
    runtime_status_sender: RuntimeStatusSender | None = None
    if relay_url.strip() and effective_relay_token:
        runtime_status_sender = RuntimeStatusSender(
            relay_url,
            effective_relay_token,
            diagnostics,
            {camera.camera_id: camera.facility_id for camera in effective_config.cameras},
        )
        runtime_status_sender.start()
    try:
        supervisor = _build_supervisor(
            effective_config,
            status_store,
            device=startup_device,
            decode=startup_decode,
            overlay_publisher=overlay_publisher,
            config_version=config_version,
            restart_check=_restart_check(relay_url, effective_relay_token, boot_registry_version),
            yaml_config=startup.yaml_config,
            pulled=startup.pulled,
            relay_url=relay_url,
            relay_token=effective_relay_token,
            clip_recorder=clip_recorder,
            diagnostics=diagnostics,
            evidence_stagers=evidence_stagers,
        )
    except (ModelLoadError, TypeError) as exc:
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if evidence_runtime is not None:
            evidence_runtime.stop_sender()
        if clip_recorder is not None:
            clip_recorder.stop()
        if runtime_status_sender is not None:
            runtime_status_sender.stop()
        print(str(exc), file=sys.stderr)
        return 2
    try:
        result = supervisor.run(
            max_frames_per_camera=options.max_frames_per_camera,
            heartbeat_on_start=options.heartbeat_on_start,
        )
    finally:
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if evidence_runtime is not None:
            evidence_runtime.stop_sender()
        if clip_recorder is not None:
            clip_recorder.stop()
        if runtime_status_sender is not None:
            runtime_status_sender.stop()
    print(
        json.dumps({"processed": result, "status": status_store.snapshot()}, separators=(",", ":"))
    )
    if options.max_frames_per_camera is not None and any(
        count < options.max_frames_per_camera for count in result.values()
    ):
        return 1
    return 0


def _parse_args(args: list[str]) -> _Options:
    config_path: str | None = None
    check_config = False
    heartbeat_on_start = False
    max_frames_per_camera: int | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config":
            index += 1
            if index >= len(args):
                raise EdgeWorkerConfigError("--config requires a path")
            config_path = args[index]
        elif arg == "--check-config":
            check_config = True
        elif arg == "--heartbeat-on-start":
            heartbeat_on_start = True
        elif arg == "--max-frames-per-camera":
            index += 1
            if index >= len(args):
                raise EdgeWorkerConfigError("--max-frames-per-camera requires a value")
            max_frames_per_camera = _positive_int(args[index], "--max-frames-per-camera")
        else:
            raise EdgeWorkerConfigError(f"unknown argument: {arg}")
        index += 1
    return _Options(
        config_path=config_path,
        check_config=check_config,
        max_frames_per_camera=max_frames_per_camera,
        heartbeat_on_start=heartbeat_on_start,
    )


def _publish_profile_boot_failure(environ: Mapping[str, str], reason: str) -> None:
    relay_url = environ.get("RELAY_URL", "").strip()
    relay_token = environ.get("RELAY_TOKEN", "").strip()
    facility_id = environ.get("API_FACILITY_ID", "").strip()
    if not relay_url or not relay_token or not facility_id:
        print(
            "Profile boot diagnostics publish failed: relay configuration is unavailable",
            file=sys.stderr,
        )
        return
    try:
        diagnostics = WorkerDiagnostics(
            gpu=_gpu_diagnostics(),
            worker={
                "alive": False,
                "pid": os.getpid(),
                "profile_boot_error": reason,
            },
        )
        sender = RuntimeStatusSender(
            relay_url,
            relay_token,
            diagnostics,
            facility_id,
            timeout_sec=_PROFILE_BOOT_STATUS_TIMEOUT_SEC,
        )
        if not sender.publish_once():
            print("Profile boot diagnostics publish failed", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not affect fail-fast exit
        print(
            f"Profile boot diagnostics publish failed: {exc.__class__.__name__}",
            file=sys.stderr,
        )


def _gpu_diagnostics() -> dict[str, object]:
    payload: dict[str, object] = {
        "nvml_available": False,
        "nvml_error": None,
        "cuda_context_ok": False,
        "driver_version": None,
        "device_name": None,
        "captured_at_sec": time.time(),
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            payload["nvml_available"] = True
            payload["driver_version"] = _nvml_text(pynvml.nvmlSystemGetDriverVersion())
            payload["device_name"] = _nvml_text(pynvml.nvmlDeviceGetName(handle))
        finally:
            pynvml.nvmlShutdown()
    except ModuleNotFoundError:
        payload["nvml_error"] = "binding_unavailable"
    except Exception as exc:  # noqa: BLE001 - diagnostics must not affect GPU fail-fast
        payload["nvml_error"] = f"query_failed:{exc.__class__.__name__}"
    try:
        import torch

        torch.cuda.init()
        payload["cuda_context_ok"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001,S110 - diagnostics must not affect GPU fail-fast
        pass
    return payload


def _nvml_text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _effective_relay_token(config: EdgeWorkerConfig, relay_token: str | None) -> str:
    env_token = relay_token.strip() if relay_token is not None else ""
    return env_token or config.relay.token.get_secret_value()


def _load_startup_config(options: _Options) -> _StartupConfig:
    if _yaml_config_requested(options):
        config = load_edge_worker_config(resolve_config_path(options.config_path))
        print(
            f"{EDGE_CAMERA_CONFIG_ENV} YAML bootstrap is configured; "
            "worker config pull is bypassed",
            file=sys.stderr,
        )
        return _StartupConfig(config=config, registry_version=0, source="yaml", yaml_config=config)

    relay_url = _required_env("RELAY_URL")
    relay_token = os.environ.get("RELAY_TOKEN")
    loaded = load_edge_worker_config_from_relay(relay_url, relay_token)
    if loaded is None:
        raise EdgeWorkerConfigError("worker config pull failed and LKG is unavailable")
    config, registry_version, source = loaded
    return _StartupConfig(config=config, registry_version=registry_version, source=source)


def _yaml_config_requested(options: _Options) -> bool:
    if options.config_path is not None:
        return True
    return bool(os.environ.get(EDGE_CAMERA_CONFIG_ENV, "").strip())


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value == "":
        raise EdgeWorkerConfigError(f"{name} is required when {EDGE_CAMERA_CONFIG_ENV} is unset")
    return value


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise EdgeWorkerConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise EdgeWorkerConfigError(f"{name} must be > 0")
    return value


def _build_supervisor(
    config: EdgeWorkerConfig,
    status_store: StatusStore,
    *,
    device: str,
    decode: str,
    registry: ModelRegistry | None = None,
    overlay_publisher: OverlayPublisher | None = None,
    config_version: int = 0,
    restart_check: Callable[[], bool] | None = None,
    yaml_config: EdgeWorkerConfig | None = None,
    pulled: PulledWorkerConfig | None = None,
    relay_url: str | None = None,
    relay_token: str | None = None,
    clip_recorder: ClipRecorder | None = None,
    diagnostics: WorkerDiagnostics | None = None,
    evidence_stagers: Mapping[str, EvidenceStagerProtocol] | None = None,
) -> EdgeWorkerSupervisor:
    model_registry = DEFAULT_REGISTRY if registry is None else registry
    serving = InProcessServingClient(model_registry)
    clients = {
        camera.camera_id: _relay_client(config, camera, config_version) for camera in config.cameras
    }
    stop_event = threading.Event()
    resources = _WorkerResources(
        clients=clients,
        runners=_build_runner_bundle(serving, device=device),
        fall_model=_build_fall_model(config, serving, device=device),
        status_store=status_store,
        config=config,
        overlay_publisher=overlay_publisher,
        stop_event=stop_event,
        clip_recorder=clip_recorder,
        diagnostics=diagnostics,
        evidence_stagers={} if evidence_stagers is None else evidence_stagers,
        decode=decode,
    )
    if clip_recorder is not None:
        for camera in config.cameras:
            clip_recorder.set_camera_fps(camera.camera_id, camera.fps)
    workers_and_detectors = tuple(
        _worker_with_detectors(camera, resources) for camera in config.cameras
    )
    workers = tuple(worker for worker, _detectors in workers_and_detectors)
    detectors = tuple(
        detector
        for _worker, worker_detectors in workers_and_detectors
        for detector in worker_detectors
    )
    night_window_source = config if yaml_config is None else yaml_config
    initial_runtime_config = resolve_runtime_config(night_window_source, pulled)
    apply_runtime_config(detectors, night_window_source, pulled)
    interval = min(camera.heartbeat_interval_sec for camera in config.cameras)
    config_refresh = None
    if relay_url is not None:
        config_refresh = _config_refresh(
            relay_url,
            relay_token,
            detectors,
            night_window_source,
            initial_runtime_config,
        )
    return EdgeWorkerSupervisor.from_workers(
        workers,
        status_store=status_store,
        heartbeat_sinks=clients,
        heartbeat_interval_sec=interval,
        stop_event=stop_event,
        restart_check=restart_check,
        config_refresh=config_refresh,
    )


def _build_runner_bundle(serving: ServingClient, *, device: str) -> _RunnerBundle:
    return _RunnerBundle(
        pose=warmup_to_ready(serving.create("pose", device=device), device=device).runner,
        bed=warmup_to_ready(serving.create("bed", device=device), device=device).runner,
    )


def _build_fall_model(
    config: EdgeWorkerConfig, serving: ServingClient, *, device: str = "cpu"
) -> FallModelProtocol:
    fall_config = config.models.fall
    if fall_config is not None:
        return LstmFallRunner.from_artifact_dir(
            fall_config.artifact_dir,
            device=device,
            expected_schema_version=fall_config.schema_version,
            expected_preprocessing_identity=fall_config.preprocessing_identity,
        )
    model_name = next(
        registration.model_name
        for registration in DOMAIN_REGISTRY.values()
        if registration.model_name is not None
    )
    model = serving.create(model_name)
    return _require_fall_model(model)


def _require_fall_model(model: RunnerProtocol) -> FallModelProtocol:
    if not isinstance(model, FallModelProtocol):
        raise TypeError("fall model must expose operating_threshold and predict")
    return model


def _decode_backend(
    camera: CameraRuntimeConfig,
    diagnostics: WorkerDiagnostics | None = None,
    *,
    profile_decode: str | None = None,
) -> RTSPBackend:
    backend_name = profile_decode if profile_decode is not None else camera.decode_backend
    if diagnostics is None:
        return create_backend(backend_name)
    return create_backend(
        backend_name,
        on_selection=lambda selection: diagnostics.update_decode(camera.camera_id, selection),
    )


def _worker_with_detectors(
    camera: CameraRuntimeConfig,
    resources: _WorkerResources,
) -> tuple[CameraWorker, tuple[DomainDetector, ...]]:
    detectors = _domain_detectors(resources.config, resources.fall_model)
    return _worker(camera, resources, domain_detectors=detectors), detectors


def _worker(
    camera: CameraRuntimeConfig,
    resources: _WorkerResources,
    *,
    domain_detectors: tuple[DomainDetector, ...] | None = None,
) -> CameraWorker:
    runtime = resources.config.runtime
    tracker = GreedyIouTracker()
    return CameraWorker(
        camera_id=camera.camera_id,
        facility_id=camera.facility_id,
        frame_source=RTSPSource(
            camera.inference_rtsp_url,
            max_failures=runtime.max_failures,
            open_timeout_ms=runtime.open_timeout_ms,
            read_timeout_ms=runtime.read_timeout_ms,
            # Production keeps retrying indefinitely so recoverable safety cameras
            # come back online; each camera has its own capture thread, reports
            # DEGRADED while down, and is promptly cancellable via stop_event.
            backoff_wait=resources.stop_event.wait if resources.stop_event is not None else None,
            stop_requested=(
                resources.stop_event.is_set if resources.stop_event is not None else None
            ),
            backend=_decode_backend(camera, resources.diagnostics, profile_decode=resources.decode),
            target_fps=camera.fps,
            pace_wait=resources.stop_event.wait if resources.stop_event is not None else None,
            on_open_failure=(
                lambda reason: (
                    resources.diagnostics.record_decode_open_failure(camera.camera_id, reason)
                    if resources.diagnostics is not None
                    else None
                )
            ),
        ),
        runners=resources.runners.as_mapping(),
        scheduler=Scheduler(
            {
                "pose": camera.frame_stride,
                "bed": max(30, camera.frame_stride),
            }
        ),
        domain_detectors=(
            _domain_detectors(resources.config, resources.fall_model)
            if domain_detectors is None
            else domain_detectors
        ),
        event_sink=resources.clients[camera.camera_id],
        status_store=resources.status_store,
        tracker=tracker,
        overlay_sink=resources.overlay_publisher,
        snapshot_renderer=OverlayRenderer(),
        detector_version=DETECTOR_VERSION,
        clip_recorder=resources.clip_recorder,
        evidence_stager=resources.evidence_stagers.get(camera.camera_id),
        event_identity_path=event_identity_path(camera.camera_id),
        diagnostics=resources.diagnostics,
    )


def _domain_config_payload(config: EdgeWorkerConfig, name: str) -> dict[str, object] | None:
    try:
        domain_config = config.domains.domain_config(name)
    except ValueError:
        return None
    if domain_config is None:
        return None
    return domain_config.model_dump(exclude={"enabled"}, exclude_none=True)


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _domain_detectors(
    config: EdgeWorkerConfig,
    fall_model: FallModelProtocol | None = None,
    *,
    audit_model: object | None = None,
) -> tuple[DomainDetector, ...]:
    enabled = config.enabled_domains
    detectors: list[DomainDetector] = []
    for name, registration in DOMAIN_REGISTRY.items():
        if not (registration.enabled if enabled is None else name in enabled):
            continue
        detector = registration.factory(_domain_config_payload(config, name), fall_model)
        model_for_audit = fall_model if audit_model is None else audit_model
        detector.audit_context = AuditContext(
            model_version=_str_or_none(getattr(model_for_audit, "version", None)),
            operating_threshold=_float_or_none(
                getattr(model_for_audit, "operating_threshold", None)
            ),
        )
        detector.registration = registration
        detectors.append(detector)
    return tuple(detectors)


@dataclass(slots=True)
class _RelayClient:
    alert_url: str
    heartbeat_url: str
    camera_id: str
    facility_id: str
    resident_id: str | None
    relay_token: str
    allowed_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            event_type
            for registration in DOMAIN_REGISTRY.values()
            for event_type in registration.event_types
        )
    )
    timeout_sec: float = 0.5
    config_version: int = 0
    failure_count: int = 0

    def __post_init__(self) -> None:
        self.alert_url = _parse_http_url(self.alert_url)
        self.heartbeat_url = _parse_http_url(self.heartbeat_url)
        self.relay_token = _required(self.relay_token, "relay_token")

    def send_heartbeat(self) -> bool:
        return self._post(
            self.heartbeat_url,
            {
                "camera_id": self.camera_id,
                "facility_id": self.facility_id,
                CONFIG_VERSION_KEY: self.config_version,
            },
        )

    def _snapshot_metadata(
        self, snapshot: object, edge_event_id: str | None
    ) -> dict[str, object] | None:

        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot metadata must be an object")
        expected_keys = {
            "snapshot_id",
            "path",
            "sha256",
            "size_bytes",
            "mime_type",
            "captured_at",
            "camera_id",
            "edge_event_id",
        }
        if set(snapshot) != expected_keys:
            raise ValueError("snapshot metadata has unexpected fields")
        text_keys = expected_keys - {"size_bytes", "edge_event_id"}
        if any(not isinstance(snapshot[key], str) or not snapshot[key] for key in text_keys):
            raise ValueError("snapshot metadata has invalid text fields")
        snapshot_id = snapshot["snapshot_id"]
        captured_at = snapshot["captured_at"]
        if snapshot["camera_id"] != self.camera_id:
            raise ValueError("snapshot camera does not match relay camera")
        if snapshot["mime_type"] != "image/jpeg":
            raise ValueError("snapshot MIME type must be image/jpeg")
        sha256 = snapshot["sha256"]
        if (
            len(sha256) != 64
            or sha256.lower() != sha256
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("snapshot sha256 must be lowercase SHA-256")
        size_bytes = snapshot["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError("snapshot size must be positive")
        if not _is_canonical_utc_timestamp(captured_at):
            raise ValueError("snapshot captured_at must be canonical UTC")
        expected_path = _snapshot_relative_path(self.camera_id, captured_at, snapshot_id)
        if snapshot["path"] != expected_path:
            raise ValueError("snapshot path does not match snapshot identity")
        if (
            edge_event_id is None
            or snapshot["edge_event_id"] != edge_event_id
            or snapshot_id != edge_event_id
        ):
            raise ValueError("snapshot identity does not match event identity")
        return dict(snapshot)

    def emit(self, event: EventPayload) -> None:
        event_type = str(event.get("event_type", ""))
        if event_type not in self.allowed_event_types:
            raise ValueError(f"unregistered domain event type: {event_type!r}")
        evidence = dict(event)
        event_audit = evidence.pop("audit", None)
        snapshot_jpeg_present = "snapshot_jpeg" in evidence
        snapshot_jpeg = evidence.pop("snapshot_jpeg", None)
        snapshot_present = "snapshot" in evidence
        snapshot_value = evidence.pop("snapshot", None)
        raw_edge_event_id = evidence.pop("edge_event_id", None)

        if raw_edge_event_id is not None and (
            not isinstance(raw_edge_event_id, str) or not _is_canonical_uuid(raw_edge_event_id)
        ):
            raise ValueError("edge_event_id must be a canonical UUID")
        edge_event_id = raw_edge_event_id
        payload: dict[str, object] = {
            "event_type": event_type,
            "probability": _event_probability(event),
            "detected_at": str(event.get("detected_at", "")) or _utc_timestamp(),
            "camera_id": self.camera_id,
            "facility_id": self.facility_id,
            "evidence": evidence,
        }
        if self.resident_id is not None:
            payload["resident_id"] = self.resident_id
        if isinstance(event_audit, dict):
            payload["audit"] = {**event_audit, CONFIG_VERSION_KEY: self.config_version}
        if snapshot_jpeg_present:
            if not isinstance(snapshot_jpeg, bytes):
                raise TypeError("snapshot_jpeg must be bytes")
            if not snapshot_jpeg or len(snapshot_jpeg) > MAX_SNAPSHOT_BYTES:
                raise ValueError("snapshot_jpeg exceeds the bounded relay contract")
            # Bytes feed the remote Event API snapshot while metadata indexes the
            # already-persisted local artifact; they are complementary contracts.
            payload["snapshot_jpeg_base64"] = base64.b64encode(snapshot_jpeg).decode("ascii")
        snapshot = (
            self._snapshot_metadata(snapshot_value, edge_event_id) if snapshot_present else None
        )
        if snapshot is not None:
            payload["snapshot"] = snapshot
        if edge_event_id is not None:
            payload["edge_event_id"] = edge_event_id
        self._post(self.alert_url, payload)

    def _post(self, url: str, payload: dict[str, object]) -> bool:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-Edge-Relay-Token": self.relay_token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response.read()
        except (TimeoutError, OSError, urllib.error.URLError, urllib.error.HTTPError):
            self.failure_count += 1
            return False
        return True


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _is_canonical_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return False
    canonical = parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value == canonical


def _snapshot_relative_path(camera_id: str, captured_at: str, snapshot_id: str) -> str:
    camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
    snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
    date = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).date().isoformat()
    return f"snapshots/{camera_key}/{date}/{snapshot_key}.jpg"


def _relay_client(
    config: EdgeWorkerConfig,
    camera: CameraRuntimeConfig,
    config_version: int = 0,
) -> _RelayClient:
    return _RelayClient(
        alert_url=config.relay_alert_url,
        heartbeat_url=config.relay_heartbeat_url,
        camera_id=camera.camera_id,
        facility_id=camera.facility_id,
        resident_id=camera.resident_id,
        relay_token=config.relay.token.get_secret_value(),
        config_version=config_version,
    )


def _restart_check(
    relay_url: str,
    relay_token: str | None,
    boot_registry_version: int,
    *,
    poll_interval_sec: float = 60.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> Callable[[], bool]:
    last_checked = -poll_interval_sec

    def _check() -> bool:
        nonlocal last_checked
        now = monotonic()
        if now - last_checked < poll_interval_sec:
            return False
        last_checked = now
        pulled = pull_worker_config(relay_url, relay_token)
        if pulled is None or pulled.config_version <= boot_registry_version:
            return False
        print(
            f"worker registry_version changed "
            f"{boot_registry_version}->{pulled.config_version}; exiting for restart",
            file=sys.stderr,
        )
        return True

    return _check


def _config_refresh(
    relay_url: str,
    relay_token: str | None,
    detectors: tuple[DomainDetector, ...],
    yaml_config: EdgeWorkerConfig,
    initial_runtime_config: Mapping[str, object | None],
    *,
    pull_config: Callable[[str, str | None], PulledWorkerConfig | None] = pull_worker_config,
) -> Callable[[], None]:
    last_applied = initial_runtime_config

    def _refresh() -> None:
        nonlocal last_applied
        try:
            pulled = pull_config(relay_url, relay_token)
            if pulled is None:
                return
            runtime_config = resolve_runtime_config(yaml_config, pulled)
            if runtime_config == last_applied:
                return
            apply_runtime_config(detectors, yaml_config, pulled)
            last_applied = runtime_config
        except Exception:  # noqa: BLE001 - config refresh is best-effort; keep last window
            return

    return _refresh


def _parse_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise ValueError(f"relay URL must be absolute HTTP(S): {url}")
    return url


def _required(value: str, name: str) -> str:
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{name} must be set")
    return stripped


def _event_probability(event: EventPayload) -> float:
    value = event.get("probability", event.get("confidence", 1.0))
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    return 1.0


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("main",)
