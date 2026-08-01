from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Final, Protocol, TypeAlias, final, runtime_checkable

import worker.pipeline.ingest.lifecycle as ingest
import worker.runtime.bootstrap as bootstrap
from contracts.runner import RunnerProtocol
from shared.events.evidence_export_contract import DeliveryFailure
from shared.events.evidence_http_transport import bounded_request, encode_json
from worker.adapters.model import LstmFallRunner, warmup_to_ready
from worker.adapters.model.errors import FatalAcceleratorError
from worker.domains import (
    DOMAIN_REGISTRY,
    BedExitDomainDependencies,
    FallDomainDependencies,
    enabled_domains,
)
from worker.domains.bed_exit import BedExitConfig, NightWindow
from worker.domains.fall import FallModelProtocol
from worker.interfaces.decision import Decider
from worker.interfaces.output import EventSink
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.ingest.registry import SourceRegistry
from worker.pipeline.output.event_sink import EventClipRecorder, EvidenceEventSink
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_services import default_services
from worker.pipeline.output.evidence.evidence_runtime import (
    OUTBOX_PATH_ENV,
    EvidenceExportRuntime,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.ingest_composition import (
    build_camera_source_registry,
    compose_camera_ingest_loop,
)
from worker.runtime.lease import GpuLease, resolve_state_dir
from worker.runtime.model_composition import SharedYoloExtractors, compose_yolo_extractors
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import DecodeProbe
from worker.runtime.watchdog import InferenceWatchdog

LOGGER: Final = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT_SEC: Final = 0.5


class _RunnableIngest(Protocol):
    @property
    def camera_id(self) -> str: ...

    def run(self) -> None: ...

    def stop(self) -> None: ...


CameraLoopFactory: TypeAlias = Callable[
    [CameraRuntimeConfig, BoundedFrameBus, ingest.IngestReporter], _RunnableIngest
]

PumpFactory: TypeAlias = Callable[
    [CameraRuntimeConfig, BoundedFrameBus, CompositeExtractor, EventAggregator, EventSink],
    _RunnableIngest,
]

EventSinkFactory: TypeAlias = Callable[[CameraRuntimeConfig], EventSink]

ClipRecorderFactory: TypeAlias = Callable[[CameraRuntimeConfig], EventClipRecorder]


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraRuntimeContext:
    bus: BoundedFrameBus
    tracker: GreedyIouTracker
    scene_state: SceneState
    scheduler: Scheduler
    analytics: CompositeExtractor
    decision: EventAggregator
    heartbeat: HeartbeatReporter
    ingest_loop: _RunnableIngest
    pump: _RunnableIngest


@final
class HeartbeatReporter:
    """Translate a camera READY transition into one bounded relay heartbeat."""

    def __init__(self, worker: WorkerConfig, camera: CameraRuntimeConfig) -> None:
        self._worker, self._camera = worker, camera
        self._last_attempt: float | None = None
        self.failure_count = 0

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
        payload = {"camera_id": camera_id, "facility_id": camera.facility_id,
                   "config_version": worker.version}
        headers = {
            "Content-Type": "application/json",
            "X-Edge-Relay-Token": worker.relay.token.get_secret_value(),
        }
        try:
            result = bounded_request(
                worker.relay_heartbeat_url, "POST", headers,
                encode_json(payload), HEARTBEAT_TIMEOUT_SEC,
            )
        except FatalAcceleratorError:
            raise
        except Exception:  # noqa: BLE001 - relay I/O is a non-fatal camera boundary
            self._record_failure()
            return
        if isinstance(result, DeliveryFailure) or not 200 <= result[0] < 300:
            self._record_failure()

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: ingest.IngestEvent) -> None:
        del event

    def _record_failure(self) -> None:
        self.failure_count += 1
        LOGGER.warning("relay heartbeat failed", extra={"camera_id": self._camera.camera_id})


@final
class _FaultAwareLoop:
    def __init__(
        self,
        loop: _RunnableIngest,
        handler: FaultHandler,
        profile: str,
        *,
        stage: str = "camera_ingest",
    ) -> None:
        self._loop, self._handler = loop, handler
        self._profile = profile
        self._stage = stage

    @property
    def camera_id(self) -> str:
        return self._loop.camera_id

    def run(self) -> None:
        try:
            self._loop.run()
        except FatalAcceleratorError as exc:
            record = make_fault_record(
                exc, profile=self._profile, task=exc.task or "inference",
                stage=self._stage, camera_id=exc.camera_id or self.camera_id,
            )
            self._handler.handle(exc, record)

    def stop(self) -> None:
        self._loop.stop()


@final
class _NullClipRecorder:
    """Interim ``EventClipRecorder``: real clip binding lands with the clip-encoder
    composition; until then, events still stage durably without a bound clip
    (the same branch :class:`EvidenceEventSink` already exercises when
    recording is unavailable).
    """

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        del camera_id, event_ref, event_type, allow_new_clip
        return None


@final
@dataclass(frozen=True, slots=True)
class _CameraClipRecorderView:
    """Per-camera ``EventClipRecorder`` view over one shared :class:`ClipRecorder`.

    ``ClipRecorder`` is a single actor (one encoder, one queue, one thread) that
    already keys all mutable per-clip state by ``camera_id`` internally, so
    clip state is never cross-contaminated between cameras even though the
    underlying encoder resource is shared -- that sharing is the existing
    design, not something introduced here. This view exists so composition
    hands each camera a distinct ``EventClipRecorder`` object (never the same
    reference), matching the DI seam style used elsewhere in this module.
    """

    recorder: ClipRecorder
    camera_id: str

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        return self.recorder.on_event(
            camera_id, event_ref, event_type, allow_new_clip=allow_new_clip
        )


def _evidence_outbox_path() -> Path:
    configured = os.environ.get(OUTBOX_PATH_ENV, "").strip()
    if configured:
        return Path(configured)
    return resolve_state_dir() / "evidence-outbox.sqlite3"


def _default_pump_factory(
    camera: CameraRuntimeConfig,
    bus: BoundedFrameBus,
    analytics: CompositeExtractor,
    decision: EventAggregator,
    sink: EventSink,
) -> CameraPipelinePump:
    return CameraPipelinePump(camera.camera_id, bus.inference, analytics, decision, sink)


@final
class WorkerRuntime:
    """Own process-wide models and camera-local mutable pipeline state."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        loop_factory: CameraLoopFactory | None = None,
        serving_client: ServingClient,
        env: Mapping[str, str] | None = None,
        acquire_lease: bootstrap.LeaseAcquirer | None = None,
        decode_probe: DecodeProbe | None = None,
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        restart_check: Callable[[], bool] | None = None,
        pump_factory: PumpFactory = _default_pump_factory,
        event_sink_factory: EventSinkFactory | None = None,
        clip_recorder_factory: ClipRecorderFactory | None = None,
    ) -> None:
        self.config = config
        self._env = os.environ if env is None else env
        self._serving = serving_client
        self._loop_factory = loop_factory or self._default_loop_factory
        self._pump_factory = pump_factory
        self._sink_factory = event_sink_factory or self._default_event_sink
        self._clip_recorder_factory = clip_recorder_factory or self._default_clip_recorder
        self._acquire = acquire_lease or GpuLease.acquire
        self._decode_probe, self._boot_dependencies = decode_probe, boot_dependencies
        self._hard_exit = hard_exit
        self._restart_check = restart_check
        self._context = bootstrap.BootstrapContext()
        self._boot: BootContext | None = None
        self._supervisor: ingest.IngestSupervisor | None = None
        self.shared_yolo: SharedYoloExtractors | None = None
        self.fall_model: FallModelProtocol | None = None
        self.fault_handler: FaultHandler | None = None
        self.watchdog: InferenceWatchdog | None = None
        self.cameras: tuple[CameraRuntimeContext, ...] = ()
        self._clip_recorder: ClipRecorder | None = None
        self._evidence_export_runtime: EvidenceExportRuntime | None = None
        self._camera_source_registry: SourceRegistry | None = None

    def run(self) -> None:
        stages = bootstrap.named_stages(
            self._context, self._env,
            initializers={"models": self._initialize_models},
            warmups={"models": lambda _models: self._warm_models()},
            activate=self._activate,
            decode_probe=self._decode_probe, deps=self._boot_dependencies,
            acquire=self._acquire,
        )
        try:
            _ = bootstrap.bootstrap_or_exit(stages, context=self._context)
            self._start_export_sender()
            if self._supervisor is not None:
                self._supervisor.join()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
        if self.watchdog is not None:
            self.watchdog.stop()
        if self._evidence_export_runtime is not None:
            self._evidence_export_runtime.stop_sender()
        if self._clip_recorder is not None:
            self._clip_recorder.stop()
        self._context.release_lease()

    def _start_export_sender(self) -> None:
        """Start delivering staged evidence to the relay, never fatal to cameras.

        Camera activation (and the clip recorder it composes) has already run
        by the time ``bootstrap_or_exit`` returns, so this only needs to flip
        the sender's background thread on.
        """
        if self._evidence_export_runtime is None:
            return
        try:
            self._evidence_export_runtime.start_sender()
        except Exception:  # noqa: BLE001 - export delivery is a non-fatal camera boundary
            LOGGER.warning("evidence export sender failed to start", exc_info=True)

    def _initialize_models(
        self, boot: BootContext
    ) -> tuple[SharedYoloExtractors, FallModelProtocol]:
        self._boot = boot
        self.fault_handler = FaultHandler(boot.profile.name, hard_exit=self._hard_exit)
        self.watchdog = InferenceWatchdog(self.fault_handler, profile=boot.profile.name)
        self.shared_yolo = compose_yolo_extractors(self._serving, device=boot.device)
        self.fall_model = self._create_fall_model(boot.device)
        return self.shared_yolo, self.fall_model

    def _create_fall_model(self, device: str) -> FallModelProtocol:
        configured = self.config.models.fall
        if configured is not None:
            return LstmFallRunner.from_artifact_dir(
                configured.artifact_dir, device=device,
                expected_schema_version=configured.schema_version,
                expected_preprocessing_identity=configured.preprocessing_identity,
            )
        model = self._serving.create("fall")
        if isinstance(model, FallModelProtocol):
            return model
        raise RuntimeError("fall model does not satisfy FallModelProtocol")

    def _warm_models(self) -> tuple[str, ...]:
        if self.shared_yolo is None or self.fall_model is None or self._boot is None:
            raise RuntimeError("models cannot warm before initialization")
        for extractor in self.shared_yolo.extractors:
            self._warm_one(extractor.runner, self._boot.device)
        self._warm_one(self.fall_model, self._boot.device)
        return ("pose", "person", "bed", "fall")

    def _warm_one(self, model: RunnerProtocol | FallModelProtocol, device: str) -> None:
        if not isinstance(model, _Warmable):
            raise TypeError("configured model does not expose warmup")
        _ = warmup_to_ready(model, device=device)

    def _activate(self, boot: BootContext) -> tuple[bootstrap.CameraStageOutcome, ...]:
        yolo, handler, watchdog = self.shared_yolo, self.fault_handler, self.watchdog
        if yolo is None or handler is None or watchdog is None:
            raise RuntimeError("camera activation requires initialized shared state")
        self._compose_evidence_export(boot)
        contexts: list[CameraRuntimeContext] = []
        outcomes: list[bootstrap.CameraStageOutcome] = []
        for camera in self.config.cameras:
            built: list[CameraRuntimeContext] = []
            outcomes.append(bootstrap.run_camera_stage(
                camera.camera_id,
                lambda camera=camera, built=built: built.append(self._build_camera(camera, yolo)),
            ))
            contexts.extend(built)
        self.cameras = tuple(contexts)
        loops = tuple(
            _FaultAwareLoop(item.ingest_loop, handler, boot.profile.name) for item in contexts
        ) + tuple(
            _FaultAwareLoop(item.pump, handler, boot.profile.name, stage="camera_pipeline_pump")
            for item in contexts
        )
        for loop in loops:
            handler.register_loop(loop)
        self._supervisor = ingest.IngestSupervisor(loops, restart_check=self._restart_check)
        watchdog.start()
        self._supervisor.start()
        return tuple(outcomes)

    def _build_camera(
        self, camera: CameraRuntimeConfig, yolo: SharedYoloExtractors
    ) -> CameraRuntimeContext:
        if self.fall_model is None:
            raise RuntimeError("camera activation requires an initialized fall model")
        bus, tracker = BoundedFrameBus(), GreedyIouTracker()
        intervals = {"pose": camera.frame_stride, "person": camera.frame_stride, "bed": 30}
        scene, scheduler = SceneState(camera.camera_id), Scheduler(intervals)
        analytics = CompositeExtractor(
            extractors=yolo.extractors, scheduler=scheduler, tracker=tracker, scene_state=scene
        )
        decision = self._build_decision_stage(camera, self.fall_model)
        heartbeat = HeartbeatReporter(self.config, camera)
        loop = self._loop_factory(camera, bus, heartbeat)
        sink = self._sink_factory(camera)
        pump = self._pump_factory(camera, bus, analytics, decision, sink)
        return CameraRuntimeContext(
            bus, tracker, scene, scheduler, analytics, decision, heartbeat, loop, pump
        )

    def _default_event_sink(self, camera: CameraRuntimeConfig) -> EventSink:
        stager = DurableEvidenceStager(
            database_path=_evidence_outbox_path(),
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            resident_id=camera.resident_id,
            config_version=self.config.version,
            clock=time.time,
        )
        return EvidenceEventSink(stager=stager, recorder=self._clip_recorder_factory(camera))

    def _compose_evidence_export(self, boot: BootContext) -> None:
        """Compose the real clip recorder and (optional) relay export sender.

        ``ClipRecorder`` is one shared actor/encoder for the whole process (the
        existing design; see ``_CameraClipRecorderView``), so it is built once
        here, before any per-camera sink needs it. Every profile in
        ``PROFILE_REGISTRY`` resolves a real encoder (h264_nvenc or libx264) --
        there is no "no encode support" profile in this codebase -- so this
        always attempts real clip recording rather than inventing a disabled
        branch. Composition never fails camera activation: clip-store or relay
        misconfiguration/unavailability degrades to ``_NullClipRecorder`` (events
        still stage durably, just without a bound clip), matching the branch
        ``EvidenceEventSink`` already exercises when recording is unavailable.
        """
        clip_config = ClipRecorderConfig()
        evidence_runtime: EvidenceExportRuntime | None = None
        try:
            evidence_runtime = EvidenceExportRuntime.from_environment(
                store_dir=clip_config.store_dir,
                relay_url=self.config.relay.url,
                relay_token=self.config.relay.token.get_secret_value(),
                probe_camera_id=self.config.cameras[0].camera_id,
                database_path=_evidence_outbox_path(),
            )
        except ValueError:
            LOGGER.warning("evidence export misconfigured; export disabled", exc_info=True)
        recorder = ClipRecorder(
            clip_config,
            services=default_services(clip_config, boot.encode),
            is_clip_held=None if evidence_runtime is None else evidence_runtime.is_clip_held,
            startup_hook=(
                None if evidence_runtime is None else evidence_runtime.initialize_under_lock
            ),
            on_clip_finalized=(
                None if evidence_runtime is None else evidence_runtime.notify_clip_finalized
            ),
        )
        try:
            recorder.start()
        except Exception:  # noqa: BLE001 - clip recording is a non-fatal camera boundary
            LOGGER.warning("clip recorder failed to start; clips disabled", exc_info=True)
            return
        self._clip_recorder = recorder
        self._evidence_export_runtime = evidence_runtime

    def _default_clip_recorder(self, camera: CameraRuntimeConfig) -> EventClipRecorder:
        if self._clip_recorder is None:
            return _NullClipRecorder()
        return _CameraClipRecorderView(self._clip_recorder, camera.camera_id)

    def _default_loop_factory(
        self,
        camera: CameraRuntimeConfig,
        bus: BoundedFrameBus,
        reporter: ingest.IngestReporter,
    ) -> _RunnableIngest:
        """Compose the real per-camera ingest loop from the boot-resolved profile.

        ``self._boot`` is set by ``_initialize_models`` before camera activation
        (and therefore before this ever runs), so the decode token is always the
        one the profile verified at boot -- never a second, parallel resolution.
        """
        if self._boot is None:
            raise RuntimeError("camera ingest composition requires a resolved boot profile")
        return compose_camera_ingest_loop(
            camera, bus, reporter,
            decode=self._boot.decode, registry=self._ingest_source_registry(),
        )

    def _ingest_source_registry(self) -> SourceRegistry:
        if self._camera_source_registry is None:
            self._camera_source_registry = build_camera_source_registry(self.config.cameras)
        return self._camera_source_registry

    def _build_decision_stage(
        self, camera: CameraRuntimeConfig, fall_model: FallModelProtocol
    ) -> EventAggregator:
        domain_names = self.config.enabled_domains
        if domain_names is None:
            domain_names = enabled_domains()
        deciders = tuple(
            self._build_decider(name, camera, fall_model) for name in domain_names
        )
        incidents = IncidentManager(identity_path=event_identity_path(camera.camera_id))
        return EventAggregator(deciders=deciders, incidents=incidents)

    def _build_decider(
        self, name: str, camera: CameraRuntimeConfig, fall_model: FallModelProtocol
    ) -> Decider:
        registration = DOMAIN_REGISTRY[name]
        if name == "fall":
            dependencies: object = FallDomainDependencies(
                model=fall_model,
                camera_id=camera.camera_id,
                facility_id=camera.facility_id,
            )
        elif name == "bed_exit":
            dependencies = BedExitDomainDependencies(
                config=self._bed_exit_config(camera),
                clock=lambda: datetime.now(UTC),
            )
        else:
            raise RuntimeError(f"unsupported domain in registry: {name}")
        return registration.factory(dependencies)

    def _bed_exit_config(self, camera: CameraRuntimeConfig) -> BedExitConfig:
        domain_config = self.config.domains.bed_exit
        night_window = None
        if domain_config is not None and domain_config.night_window is not None:
            configured = domain_config.night_window
            night_window = NightWindow(
                start=configured.start, end=configured.end, tz=configured.tz
            )
        return BedExitConfig(
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            night_window=night_window,
        )

__all__ = ["CameraRuntimeContext", "HeartbeatReporter", "WorkerRuntime"]
