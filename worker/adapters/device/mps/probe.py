from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from shared.boundary import degrade


@dataclass(frozen=True, slots=True)
class MpsCapability:
    available: bool
    reason: str
    is_built: bool = False


TorchImporter: TypeAlias = Callable[[], Any]


def _import_torch() -> Any:
    import torch

    return torch


def probe_mps_capability(*, importer: TorchImporter = _import_torch) -> MpsCapability:
    try:
        torch = importer()
    except Exception as exc:  # noqa: BLE001
        return MpsCapability(available=False, reason=f"torch import failed: {type(exc).__name__}")

    is_built = degrade(
        lambda: bool(torch.backends.mps.is_built()), stage="mps_is_built", default=False
    )

    try:
        available = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        return MpsCapability(
            available=False,
            reason=f"torch.backends.mps.is_available() raised {type(exc).__name__}",
            is_built=is_built,
        )

    if available:
        return MpsCapability(available=True, reason="mps available", is_built=is_built)

    if not is_built:
        reason = (
            "MPS not usable: torch.backends.mps.is_built() is False -- this torch "
            "install was not built with Metal Performance Shaders support"
        )
    else:
        reason = (
            "torch.backends.mps.is_available() is False despite MPS being built into "
            "this torch install -- no usable Metal device found on this host"
        )
    return MpsCapability(available=False, reason=reason, is_built=is_built)


__all__ = ["MpsCapability", "TorchImporter", "probe_mps_capability"]
