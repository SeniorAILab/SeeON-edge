"""Runner warmup + readiness gate (ADR-0002).

``warmup_runner`` stays backward-compatible (best-effort ``warmup()`` call,
returns the runner). ``warmup_to_ready`` is the fail-fast readiness gate used by
the global bootstrap stage: it runs the runner's real warmup, forces a CUDA
sync when on a CUDA device, and only then reports ``ready``. A no-op warmup left
the first real inference to block on lazy CUDA context init (the "warmup 후 hang"
symptom); synchronizing here surfaces that cost/deadlock at startup where
fail-fast can act on it instead of hanging the worker loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def warmup_runner(runner: object) -> object:
    """Trigger a runner's cheap warmup hook when present (legacy, best-effort)."""
    warmup = getattr(runner, "warmup", None)
    if callable(warmup):
        result: Any = warmup()
        return runner if result is None else result
    return runner


def _default_cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


@dataclass(frozen=True)
class WarmupResult:
    runner: object
    ready: bool
    synchronized: bool


def warmup_to_ready(
    runner: object,
    *,
    device: str | None = None,
    synchronize: Callable[[], None] | None = None,
) -> WarmupResult:
    """Run warmup and gate readiness; raise on warmup failure (bootstrap fail-fast).

    - Calls the runner's ``warmup()`` (a real forward pass owned by the runner).
    - On a CUDA device, forces ``torch.cuda.synchronize()`` so any deferred CUDA
      init/kernel-launch cost or deadlock surfaces here, not on the first frame.
    - Returns ``ready=True`` only after both steps succeed. A ``warmup()``
      exception propagates so the global bootstrap stage can fail fast (non-zero
      exit) rather than start the worker loop in a half-initialized state.
    """
    warmed = warmup_runner(runner)

    synchronized = False
    if device is not None and str(device).startswith("cuda"):
        sync = synchronize or _default_cuda_synchronize
        sync()
        synchronized = True

    return WarmupResult(runner=warmed, ready=True, synchronized=synchronized)


__all__ = ["WarmupResult", "warmup_runner", "warmup_to_ready"]
