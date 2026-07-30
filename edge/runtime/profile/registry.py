"""Immutable device profile definitions and their local verification probes."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from edge.runners.device import probe_cuda

ML_WORKER_PROFILE_ENV = "ML_WORKER_PROFILE"


class ProfileError(RuntimeError):
    """The requested worker profile is missing or unknown."""


class ProfileVerifyError(RuntimeError):
    """A requested worker profile cannot be used on this host."""


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    profile: str
    stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: str
    device: str
    decode: str


PROFILE_REGISTRY: dict[str, ProfileSpec] = {
    "cuda": ProfileSpec("cuda", "cuda", "nvdec"),
    "mps": ProfileSpec("mps", "mps", "opencv"),
    "cpu": ProfileSpec("cpu", "cpu", "opencv"),
}


def _verify_cuda() -> VerifyResult:
    probe = probe_cuda()
    return VerifyResult(
        ok=probe.available,
        profile="cuda",
        stage="device",
        reason=probe.reason,
    )


def _verify_mps() -> VerifyResult:
    try:
        import torch

        available = torch.backends.mps.is_available()
    except Exception as exc:  # noqa: BLE001 - report unavailable runtime as structured result
        return VerifyResult(False, "mps", "device", str(exc))
    return VerifyResult(
        ok=available,
        profile="mps",
        stage="device",
        reason="MPS is available" if available else "MPS is unavailable",
    )


def _verify_cpu() -> VerifyResult:
    return VerifyResult(True, "cpu", "device", "CPU is available")


@dataclass(frozen=True, slots=True)
class BootDependencies:
    verifiers: Mapping[str, Callable[[], VerifyResult]]


def default_verifiers() -> dict[str, Callable[[], VerifyResult]]:
    return {"cuda": _verify_cuda, "mps": _verify_mps, "cpu": _verify_cpu}


def _run_ffmpeg(args: tuple[str, ...]) -> str:
    """Run a bounded ffmpeg capability query and return all reported output."""
    completed = subprocess.run(
        ("ffmpeg", "-hide_banner", *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def _nvdec_capability() -> VerifyResult:
    """Verify that ffmpeg exposes a CUDA/CUVID hardware decode capability."""
    try:
        hwaccels = _run_ffmpeg(("-hwaccels",))
    except FileNotFoundError:
        return VerifyResult(False, "cuda", "decode", "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return VerifyResult(False, "cuda", "decode", "ffmpeg capability probe timed out")
    except subprocess.CalledProcessError as exc:
        return VerifyResult(
            False,
            "cuda",
            "decode",
            f"ffmpeg capability probe failed with exit code {exc.returncode}",
        )
    except Exception as exc:  # noqa: BLE001 - decode probe must never break startup
        return VerifyResult(
            False,
            "cuda",
            "decode",
            f"ffmpeg capability probe failed: {type(exc).__name__}",
        )

    if any(capability in hwaccels.lower() for capability in ("cuda", "cuvid", "nvdec")):
        return VerifyResult(True, "cuda", "decode", "ffmpeg CUDA/CUVID hwaccel is available")

    try:
        decoders = _run_ffmpeg(("-decoders",))
    except FileNotFoundError:
        return VerifyResult(False, "cuda", "decode", "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return VerifyResult(False, "cuda", "decode", "ffmpeg decoder probe timed out")
    except subprocess.CalledProcessError as exc:
        return VerifyResult(
            False,
            "cuda",
            "decode",
            f"ffmpeg decoder probe failed with exit code {exc.returncode}",
        )
    except Exception as exc:  # noqa: BLE001 - decode probe must never break startup
        return VerifyResult(
            False,
            "cuda",
            "decode",
            f"ffmpeg decoder probe failed: {type(exc).__name__}",
        )

    if re.search(r"\b[\w-]+_cuvid\b", decoders, re.IGNORECASE):
        return VerifyResult(True, "cuda", "decode", "ffmpeg CUVID decoder is available")
    return VerifyResult(
        False,
        "cuda",
        "decode",
        "ffmpeg has no CUDA/CUVID/NVDEC hwaccel or *_cuvid decoder",
    )

def default_decode_probe(decode: str) -> VerifyResult:
    if decode == "opencv":
        try:
            import cv2  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - dependency probe reports structured failure
            return VerifyResult(False, "", "decode", str(exc))
        return VerifyResult(True, "", "decode", "OpenCV is available")
    if decode == "nvdec":
        probe = probe_cuda()
        if not probe.available:
            return VerifyResult(False, "cuda", "decode", probe.reason)
        return _nvdec_capability()
    return VerifyResult(False, "", "decode", f"unknown decode backend: {decode}")
