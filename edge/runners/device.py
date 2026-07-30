"""Device selection and GPU probing for model runners.

Two policies live here, and neither terminates the process (fail-fast ``sys.exit``
is owned by the composition-root bootstrap, per ADR-0002):

- ``select_device`` is the legacy, backward-compatible conservative selector: it
  degrades to ``"cpu"`` when torch/CUDA is unavailable. Existing callers keep
  this behavior.
- ``probe_cuda`` / ``require_gpu_device`` are the fail-fast policy: they surface
  *why* CUDA is unavailable and raise :class:`GpuUnavailableError` when a GPU was
  expected but CUDA cannot be used, instead of silently masking a broken install
  behind a CPU fallback. ``pipeline_bootstrap`` calls these at the global GPU
  verify stage and converts the exception into a non-zero process exit.

Explicit ``device="cpu"`` (operator opt-in) always wins and never fails fast.
"""

from __future__ import annotations

from dataclasses import dataclass

_CPU = "cpu"
_CUDA = "cuda"


class GpuUnavailableError(RuntimeError):
    """Raised when a GPU was expected but CUDA cannot be initialized/used."""


@dataclass(frozen=True)
class CudaProbe:
    """Result of probing torch/CUDA availability.

    ``available`` is the ground truth (a GPU can actually run inference). The
    other fields carry the diagnosis used by fail-fast error messages and by the
    Root-Cause Diagnosis in ADR-0002 (an empty ``arch_list`` means the installed
    torch wheel has no compiled device kernels — e.g. missing Blackwell sm_120).
    """

    available: bool
    reason: str
    device_count: int = 0
    arch_list: tuple[str, ...] = ()


def probe_cuda() -> CudaProbe:
    """Probe torch/CUDA without raising. Never terminates the process.

    Distinguishes the common failure modes so callers can report a precise,
    non-silent reason instead of a bare ``"cpu"``:
    - torch missing,
    - torch present but CUDA runtime init fails,
    - CUDA reports available but the build has no compiled arch kernels.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency boundary
        return CudaProbe(available=False, reason=f"torch import failed: {type(exc).__name__}")

    try:
        arch_list = tuple(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001,S110 - arch probe must not break startup
        arch_list = ()

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001,S110
        device_count = 0

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - backend probe must not break startup
        return CudaProbe(
            available=False,
            reason=f"torch.cuda.is_available() raised {type(exc).__name__}",
            device_count=device_count,
            arch_list=arch_list,
        )

    if available:
        return CudaProbe(
            available=True,
            reason="cuda available",
            device_count=device_count,
            arch_list=arch_list,
        )

    # GPU visible via NVML (device_count>0) but not usable, and no compiled
    # kernels: this is the ADR-0002 root cause (broken/CPU-only torch wheel).
    if device_count > 0 and not arch_list:
        reason = (
            f"CUDA not usable: {device_count} device(s) visible but torch build has no "
            "compiled arch kernels (empty arch_list) — likely a torch wheel without the "
            "required GPU architecture (e.g. Blackwell sm_120); see ADR-0002"
        )
    elif device_count == 0:
        reason = "torch.cuda.is_available() is False and no CUDA devices are visible"
    else:
        reason = (
            f"torch.cuda.is_available() is False despite {device_count} device(s) visible "
            f"(arch_list={list(arch_list)})"
        )
    return CudaProbe(available=False, reason=reason, device_count=device_count, arch_list=arch_list)


def select_device(explicit_device: str | None = None) -> str:
    """Return the composition-root inference device (legacy, conservative).

    An explicit device from tests/config wins. Auto-selection degrades to
    ``"cpu"`` when torch is unavailable or its backend probes fail. Callers that
    must not silently run on CPU use :func:`require_gpu_device` instead.
    """
    if explicit_device is not None:
        return explicit_device
    return _CUDA if probe_cuda().available else _CPU


def require_gpu_device(explicit_device: str | None = None) -> str:
    """Fail-fast device policy for the global GPU verify stage.

    - An explicit device wins and is returned as-is; explicit ``"cpu"``/``"opencv"``
      operator opt-in is honored without failing fast.
    - Otherwise require a usable CUDA device: return ``"cuda"`` when available, or
      raise :class:`GpuUnavailableError` with a precise reason. This never returns
      ``"cpu"`` implicitly — silent GPU→CPU degradation is the anti-pattern this
      policy removes (ADR-0002).
    """
    if explicit_device is not None:
        return explicit_device

    probe = probe_cuda()
    if probe.available:
        return _CUDA
    raise GpuUnavailableError(probe.reason)
