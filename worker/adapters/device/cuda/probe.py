from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class CudaCapability:
    available: bool
    reason: str
    device_count: int = 0
    arch_list: tuple[str, ...] = ()


TorchImporter: TypeAlias = Callable[[], Any]


def _import_torch() -> Any:
    import torch

    return torch


def probe_cuda_capability(*, importer: TorchImporter = _import_torch) -> CudaCapability:
    try:
        torch = importer()
    except Exception as exc:  # noqa: BLE001
        return CudaCapability(available=False, reason=f"torch import failed: {type(exc).__name__}")

    try:
        arch_list = tuple(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        arch_list = ()

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        device_count = 0

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        return CudaCapability(
            available=False,
            reason=f"torch.cuda.is_available() raised {type(exc).__name__}",
            device_count=device_count,
            arch_list=arch_list,
        )

    if available:
        return CudaCapability(
            available=True,
            reason="cuda available",
            device_count=device_count,
            arch_list=arch_list,
        )

    if device_count > 0 and not arch_list:
        reason = (
            f"CUDA not usable: {device_count} device(s) visible but torch build has no "
            "compiled arch kernels (empty arch_list) -- likely a torch wheel without the "
            "required GPU architecture (e.g. Blackwell sm_120)"
        )
    elif device_count == 0:
        reason = "torch.cuda.is_available() is False and no CUDA devices are visible"
    else:
        reason = (
            f"torch.cuda.is_available() is False despite {device_count} device(s) visible "
            f"(arch_list={list(arch_list)})"
        )
    return CudaCapability(
        available=False, reason=reason, device_count=device_count, arch_list=arch_list
    )


_NVENC_ENCODER_PATTERN: Final = re.compile(r"\bh264_nvenc\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NvencCapability:
    available: bool
    reason: str


class FfmpegQueryRunner(Protocol):
    def __call__(self, args: tuple[str, ...], timeout_sec: float, /) -> str: ...


def run_ffmpeg_encoders_query(args: tuple[str, ...], timeout_sec: float) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def probe_nvenc_capability(
    ffmpeg_bin: str = "ffmpeg",
    *,
    timeout_sec: float = 3.0,
    runner: FfmpegQueryRunner = run_ffmpeg_encoders_query,
) -> NvencCapability:
    try:
        encoders = runner((ffmpeg_bin, "-hide_banner", "-encoders"), timeout_sec)
    except FileNotFoundError:
        return NvencCapability(False, "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return NvencCapability(False, "ffmpeg encoder probe timed out")
    except subprocess.CalledProcessError as error:
        return NvencCapability(
            False, f"ffmpeg encoder probe failed with exit code {error.returncode}"
        )
    except Exception as error:  # noqa: BLE001
        return NvencCapability(False, f"ffmpeg encoder probe failed: {type(error).__name__}")

    if _NVENC_ENCODER_PATTERN.search(encoders):
        return NvencCapability(True, "ffmpeg h264_nvenc encoder is available")
    return NvencCapability(False, "ffmpeg has no h264_nvenc encoder")


__all__ = [
    "CudaCapability",
    "FfmpegQueryRunner",
    "NvencCapability",
    "TorchImporter",
    "probe_cuda_capability",
    "probe_nvenc_capability",
    "run_ffmpeg_encoders_query",
]
