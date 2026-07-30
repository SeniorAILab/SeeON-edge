"""Staged GPU pipeline bootstrap (ADR-0002).

Two tiers with distinct failure blast radii:

- **Global stages** run once at process start: GPU/CUDA verify -> inference
  backend init -> warmup. Any global-stage failure is fatal — the process exits
  non-zero (``bootstrap_or_exit``). There is no silent CPU fallback.
- **Per-camera stages** run per camera (NVDEC init -> worker loop). A per-camera
  failure degrades only that camera (``run_camera_stage`` returns a failed
  outcome); the process and other cameras keep running.

Stages are dependency-injected callables so the ordering/gating logic is unit
testable without a GPU or live cameras. ``sys.exit`` lives here (the composition
root), not in ``edge/runners/device.py`` (which only probes / raises).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from edge.runners.device import GpuUnavailableError, require_gpu_device

LOGGER = logging.getLogger(__name__)

# Conventional non-zero exit code for a required-infra bootstrap failure.
BOOTSTRAP_EXIT_CODE = 70


class GlobalBootstrapError(RuntimeError):
    """A required global bootstrap stage failed; the process must not continue."""


@dataclass(frozen=True)
class Stage:
    """A named global bootstrap stage. ``run`` returns the stage's output or raises."""

    name: str
    run: Callable[[], object]


@dataclass
class GlobalBootstrapResult:
    outputs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraStageOutcome:
    camera_id: str
    ok: bool
    reason: str | None = None


def gpu_verify_stage(explicit_device: str | None = None) -> Stage:
    """Global stage: require a usable inference device (fail-fast, no CPU masking)."""

    def _run() -> str:
        return require_gpu_device(explicit_device)

    return Stage(name="gpu_verify", run=_run)


def run_global_bootstrap(stages: Iterable[Stage]) -> GlobalBootstrapResult:
    """Run global stages in order; raise :class:`GlobalBootstrapError` on first failure.

    The offending stage name and cause are preserved so the caller can log a
    precise, non-silent reason.
    """
    result = GlobalBootstrapResult()
    for stage in stages:
        try:
            result.outputs[stage.name] = stage.run()
        except GpuUnavailableError as exc:
            raise GlobalBootstrapError(f"global stage {stage.name!r} failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any global-stage failure is fatal
            raise GlobalBootstrapError(f"global stage {stage.name!r} failed: {exc}") from exc
    return result


def bootstrap_or_exit(
    stages: Iterable[Stage],
    *,
    exit_fn: Callable[[int], None] = sys.exit,
    log: logging.Logger = LOGGER,
) -> GlobalBootstrapResult:
    """Run global bootstrap; on failure log loudly and exit non-zero (no fallback)."""
    try:
        return run_global_bootstrap(stages)
    except GlobalBootstrapError as exc:
        log.critical("GPU pipeline bootstrap failed; exiting (no CPU fallback): %s", exc)
        exit_fn(BOOTSTRAP_EXIT_CODE)
        raise  # pragma: no cover - reachable only if exit_fn does not terminate


def run_camera_stage(camera_id: str, run: Callable[[], object]) -> CameraStageOutcome:
    """Run a per-camera stage; a failure degrades only this camera (never exits)."""
    try:
        run()
        return CameraStageOutcome(camera_id=camera_id, ok=True)
    except Exception as exc:  # noqa: BLE001 - per-camera failure soft-degrades one camera
        LOGGER.warning("camera %s stage failed; degrading this camera only: %s", camera_id, exc)
        return CameraStageOutcome(camera_id=camera_id, ok=False, reason=str(exc))


__all__ = [
    "BOOTSTRAP_EXIT_CODE",
    "CameraStageOutcome",
    "GlobalBootstrapError",
    "GlobalBootstrapResult",
    "Stage",
    "bootstrap_or_exit",
    "gpu_verify_stage",
    "run_camera_stage",
    "run_global_bootstrap",
]
