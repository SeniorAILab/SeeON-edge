from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class NvmlGpuStatus:
    nvml_available: bool
    reason: str
    driver_version: str | None = None
    device_name: str | None = None


NvmlImporter: TypeAlias = Callable[[], Any]


def _import_pynvml() -> Any:
    import pynvml

    return pynvml


def probe_nvml_gpu_status(*, importer: NvmlImporter = _import_pynvml) -> NvmlGpuStatus:
    try:
        pynvml = importer()
    except Exception as exc:  # noqa: BLE001
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"pynvml import failed: {type(exc).__name__}: {exc}",
        )

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"nvmlInit failed: {type(exc).__name__}: {exc}",
        )

    try:
        return _read_gpu_status(pynvml)
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _read_gpu_status(pynvml: Any) -> NvmlGpuStatus:
    driver_version = _read_driver_version(pynvml)

    try:
        device_count = int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:  # noqa: BLE001
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"nvmlDeviceGetCount failed: {type(exc).__name__}: {exc}",
            driver_version=driver_version,
        )

    if device_count <= 0:
        return NvmlGpuStatus(
            nvml_available=False,
            reason="NVML initialized but no GPU devices are visible",
            driver_version=driver_version,
        )

    device_name = _read_first_device_name(pynvml)
    return NvmlGpuStatus(
        nvml_available=True,
        reason="NVML reports a usable GPU device",
        driver_version=driver_version,
        device_name=device_name,
    )


def _read_driver_version(pynvml: Any) -> str | None:
    try:
        return str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:  # noqa: BLE001
        return None


def _read_first_device_name(pynvml: Any) -> str | None:
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return str(pynvml.nvmlDeviceGetName(handle))
    except Exception:  # noqa: BLE001
        return None


__all__ = ["NvmlGpuStatus", "NvmlImporter", "probe_nvml_gpu_status"]
