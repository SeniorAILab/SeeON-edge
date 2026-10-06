from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import (
    BootContext,
    effective_decode_policy,
    effective_encode_policy,
    preflight_decode_or_raise,
    reject_legacy_conflicts,
    resolve_decode_or_fallback,
    resolve_encode_or_fallback,
    verify_device_or_raise,
)
from worker.runtime.profile.registry import (
    BootDependencies,
    DecodeProbe,
    EncodeProbe,
    ProfileSpec,
    default_decode_probe,
    default_verifiers,
    select_profile,
)

LOGGER: Final = logging.getLogger(__name__)

GPU_LEASE_STAGE: Final = "gpu_lease"
PROFILE_DEVICE_STAGE: Final = "profile_device"
DECODE_CAPABILITY_STAGE: Final = "decode_capability"
MODEL_BACKEND_INIT_STAGE: Final = "model_backend_init"
WARMUP_STAGE: Final = "real_warmup"
CAMERA_ACTIVATION_STAGE: Final = "camera_activation"

GENERIC_RUNTIME_EXIT_CODE: Final = 1
REFUSE_TO_START_EXIT_CODE: Final = 3

BackendInitializer = Callable[[BootContext], object]
BackendWarmup = Callable[[object], object]
LeaseAcquirer = Callable[[], GpuLease]


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    run: Callable[[], object]
    exit_code: int = GENERIC_RUNTIME_EXIT_CODE


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    outputs: dict[str, object]


@dataclass(frozen=True, slots=True)
class CameraStageOutcome:
    camera_id: str
    ok: bool
    reason: str | None = None


@final
class BootstrapStageError(RuntimeError):
    __slots__ = ("exit_code", "stage")

    def __init__(self, stage: str, exit_code: int, reason: str) -> None:
        super().__init__(f"bootstrap stage {stage!r} failed: {reason}")
        self.stage = stage
        self.exit_code = exit_code


@dataclass(slots=True)
class BootstrapContext:
    lease: GpuLease | None = None
    profile: ProfileSpec | None = None
    requested_profile: str | None = None
    boot: BootContext | None = None
    runners: dict[str, object] = field(default_factory=dict)
    warmed: tuple[str, ...] = ()
    warmup_complete: bool = False
    cameras: tuple[CameraStageOutcome, ...] = ()

    def release_lease(self) -> None:
        lease = self.lease
        if lease is None:
            return
        self.lease = None
        lease.close()


def run_stages(stages: Iterable[Stage]) -> BootstrapResult:
    outputs: dict[str, object] = {}
    for stage in stages:
        try:
            outputs[stage.name] = _run_stage(stage, outputs)
        except BootstrapStageError:
            raise
        except FatalAcceleratorError as exc:
            raise BootstrapStageError(
                stage.name,
                FATAL_ACCELERATOR_EXIT_CODE,
                str(exc) or type(exc).__name__,
            ) from exc
        except Exception as exc:
            raise BootstrapStageError(
                stage.name,
                stage.exit_code,
                str(exc) or type(exc).__name__,
            ) from exc
    return BootstrapResult(outputs)


def _run_stage(stage: Stage, outputs: dict[str, object]) -> object:
    parameters = inspect.signature(stage.run).parameters.values()
    takes_outputs = any(
        parameter.kind
        in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL}
        for parameter in parameters
    )
    if takes_outputs:
        run_with_outputs = cast(Callable[[dict[str, object]], object], stage.run)
        return run_with_outputs(outputs)
    return stage.run()


def bootstrap_or_exit(
    stages: Iterable[Stage],
    *,
    context: BootstrapContext | None = None,
    exit_fn: Callable[[int], None] = sys.exit,
    log: logging.Logger = LOGGER,
) -> BootstrapResult:
    try:
        return run_stages(stages)
    except BootstrapStageError as exc:
        log.critical(
            "bootstrap stage %r failed; activating zero cameras and exiting %d: %s",
            exc.stage,
            exc.exit_code,
            exc,
        )
        if context is not None:
            context.release_lease()
        exit_fn(exc.exit_code)
        raise


def gpu_lease_stage(
    context: BootstrapContext,
    *,
    acquire: LeaseAcquirer = GpuLease.acquire,
) -> Stage:
    def _run() -> object:
        lease = acquire()
        context.lease = lease
        return lease

    return Stage(GPU_LEASE_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def profile_device_stage(
    context: BootstrapContext,
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
) -> Stage:
    def _run() -> object:
        selection = select_profile(env)
        spec = selection.spec
        _ = verify_device_or_raise(spec, deps or BootDependencies(default_verifiers()))
        context.profile = spec
        context.requested_profile = selection.requested_name
        return spec

    return Stage(PROFILE_DEVICE_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def decode_capability_stage(
    context: BootstrapContext,
    env: Mapping[str, str],
    decode_probe: DecodeProbe | None = None,
    encode_probe: EncodeProbe | None = None,
) -> Stage:
    def _run() -> object:
        spec = context.profile
        if spec is None:
            raise BootstrapStageError(
                DECODE_CAPABILITY_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                "requires a completed profile/device stage",
            )
        decode_selection = None
        if spec.decode_fallback is not None:
            decode_selection = resolve_decode_or_fallback(spec, decode_probe)
        else:
            _ = preflight_decode_or_raise(spec, decode_probe or default_decode_probe)
        reject_legacy_conflicts(spec, env)
        encode_selection = resolve_encode_or_fallback(spec, encode_probe)
        degraded_reasons = tuple(
            reason
            for reason in (
                decode_selection.last_reason if decode_selection else None,
                encode_selection.last_reason,
            )
            if reason is not None
        )
        boot = BootContext(
            profile=spec,
            device=spec.device,
            decode=effective_decode_policy(spec, decode_selection),
            encode=effective_encode_policy(spec, encode_selection),
            requested_profile=context.requested_profile or spec.name,
            degraded_reasons=degraded_reasons,
            encode_selection=encode_selection,
            decode_selection=decode_selection,
        )
        context.boot = boot
        return boot

    return Stage(DECODE_CAPABILITY_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def model_backend_init_stage(
    context: BootstrapContext,
    initializers: Mapping[str, BackendInitializer],
) -> Stage:
    def _run() -> object:
        boot = context.boot
        if boot is None:
            raise BootstrapStageError(
                MODEL_BACKEND_INIT_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                f"requires a completed {DECODE_CAPABILITY_STAGE!r} stage",
            )
        LOGGER.info(
            "%s: constructing %d model backend(s): %s",
            MODEL_BACKEND_INIT_STAGE,
            len(initializers),
            ", ".join(initializers) or "(none)",
        )
        runners = {task: initializer(boot) for task, initializer in initializers.items()}
        LOGGER.info("%s: all model backends constructed", MODEL_BACKEND_INIT_STAGE)
        context.runners = runners
        return runners

    return Stage(MODEL_BACKEND_INIT_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def warmup_stage(
    context: BootstrapContext,
    warmups: Mapping[str, BackendWarmup],
) -> Stage:
    def _run() -> object:
        missing = tuple(task for task in context.runners if task not in warmups)
        unexpected = tuple(task for task in warmups if task not in context.runners)
        if missing or unexpected:
            raise BootstrapStageError(
                WARMUP_STAGE,
                GENERIC_RUNTIME_EXIT_CODE,
                f"initializer and warmup task sets differ (missing={missing}, "
                f"unexpected={unexpected})",
            )
        warmed: list[str] = []
        for task, warmup in warmups.items():
            runner = context.runners.get(task)
            if runner is None:
                raise BootstrapStageError(
                    WARMUP_STAGE,
                    GENERIC_RUNTIME_EXIT_CODE,
                    f"no initialized model backend for warmup task {task!r}",
                )
            _ = warmup(runner)
            warmed.append(task)
            context.warmed = tuple(warmed)
        context.warmup_complete = True
        return tuple(warmed)

    return Stage(WARMUP_STAGE, _run)


def camera_activation_stage(
    context: BootstrapContext,
    activate: Callable[[BootContext], Iterable[CameraStageOutcome]],
) -> Stage:
    def _run() -> object:
        if not context.warmup_complete:
            raise BootstrapStageError(
                CAMERA_ACTIVATION_STAGE,
                GENERIC_RUNTIME_EXIT_CODE,
                f"requires a completed {WARMUP_STAGE!r} stage",
            )
        boot = context.boot
        if boot is None:
            raise BootstrapStageError(
                CAMERA_ACTIVATION_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                f"requires a completed {DECODE_CAPABILITY_STAGE!r} stage",
            )
        outcomes = tuple(activate(boot))
        context.cameras = outcomes
        return outcomes

    return Stage(CAMERA_ACTIVATION_STAGE, _run)


def named_stages(
    context: BootstrapContext,
    env: Mapping[str, str],
    *,
    initializers: Mapping[str, BackendInitializer],
    warmups: Mapping[str, BackendWarmup],
    activate: Callable[[BootContext], Iterable[CameraStageOutcome]],
    decode_probe: DecodeProbe | None = None,
    encode_probe: EncodeProbe | None = None,
    deps: BootDependencies | None = None,
    acquire: LeaseAcquirer | None = None,
    acquire_lease: LeaseAcquirer | None = None,
) -> tuple[Stage, ...]:
    if acquire is not None and acquire_lease is not None:
        message = "pass either acquire or acquire_lease, not both"
        raise ValueError(message)
    resolved = acquire or acquire_lease or GpuLease.acquire
    return (
        gpu_lease_stage(context, acquire=resolved),
        profile_device_stage(context, env, deps),
        decode_capability_stage(context, env, decode_probe, encode_probe),
        model_backend_init_stage(context, initializers),
        warmup_stage(context, warmups),
        camera_activation_stage(context, activate),
    )


def run_camera_stage(camera_id: str, run: Callable[[], object]) -> CameraStageOutcome:
    try:
        _ = run()
    except FatalAcceleratorError:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "camera %s stage failed; degrading this camera only: %s",
            camera_id,
            exc,
        )
        return CameraStageOutcome(camera_id=camera_id, ok=False, reason=str(exc))
    return CameraStageOutcome(camera_id=camera_id, ok=True)


__all__ = [
    "CAMERA_ACTIVATION_STAGE",
    "DECODE_CAPABILITY_STAGE",
    "FATAL_ACCELERATOR_EXIT_CODE",
    "GENERIC_RUNTIME_EXIT_CODE",
    "GPU_LEASE_STAGE",
    "MODEL_BACKEND_INIT_STAGE",
    "PROFILE_DEVICE_STAGE",
    "REFUSE_TO_START_EXIT_CODE",
    "WARMUP_STAGE",
    "BackendInitializer",
    "BackendWarmup",
    "BootDependencies",
    "BootstrapContext",
    "BootstrapResult",
    "BootstrapStageError",
    "CameraStageOutcome",
    "LeaseAcquirer",
    "Stage",
    "bootstrap_or_exit",
    "camera_activation_stage",
    "decode_capability_stage",
    "gpu_lease_stage",
    "model_backend_init_stage",
    "named_stages",
    "profile_device_stage",
    "run_camera_stage",
    "run_stages",
    "warmup_stage",
]
