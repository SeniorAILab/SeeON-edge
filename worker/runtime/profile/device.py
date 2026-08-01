from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class CudaProbe:
    available: bool
    reason: str
    device_count: int = 0
    arch_list: tuple[str, ...] = ()


CudaProbeSource: TypeAlias = Callable[[], CudaProbe]


def probe_cuda(source: CudaProbeSource | None = None) -> CudaProbe:
    """Return an injected CUDA probe result, failing closed when absent."""
    if source is None:
        return CudaProbe(available=False, reason="CUDA capability probe is not configured")
    return source()


__all__ = ["CudaProbe", "CudaProbeSource", "probe_cuda"]
