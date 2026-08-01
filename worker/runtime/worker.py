from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.lease import GpuLease
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
    def __init__(self, loop: _RunnableIngest, handler: FaultHandler, profile: str) -> None:
        self._loop, self._handler = loop, handler
        self._profile = profile

    @property
    def camera_id(self) -> str:
        return self._loop.camera_id

    def run(self) -> None:
        try:
            self._loop.run()
        except FatalAcceleratorError as exc:
            record = make_fault_record(
                exc, profile=self._profile, task=exc.task or "inference",
                stage="camera_ingest", camera_id=exc.camera_id or self.camera_id,
            )
            self._handler.handle(exc, record)

    def stop(self) -> None:
        self._loop.stop()


@final
class WorkerRuntime:
    """Own process-wide models and camera-local mutable pipeline state."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        loop_factory: CameraLoopFactory,
        serving_client: ServingClient,
        env: Mapping[str, str] | None = None,
        acquire_lease: bootstrap.LeaseAcquirer | None = None,
        decode_probe: DecodeProbe | None = None,
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        restart_check: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self._env = os.environ if env is None else env
        self._serving = serving_client
        self._loop_factory = loop_factory
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
            if self._supervisor is not None:
                self._supervisor.join()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
        if self.watchdog is not None:
            self.watchdog.stop()
        self._context.release_lease()

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
        return CameraRuntimeContext(
            bus, tracker, scene, scheduler, analytics, decision, heartbeat, loop
        )

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
