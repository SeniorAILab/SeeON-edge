from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from worker.runtime.profile.device import CudaProbeSource, probe_cuda

DevicePolicy: TypeAlias = Literal["cuda", "mps", "cpu"]
DecodePolicy: TypeAlias = Literal["nvdec", "opencv"]
EncodePolicy: TypeAlias = Literal["h264_nvenc", "libx264"]
MpsProbeSource: TypeAlias = Callable[[], bool]

ML_WORKER_PROFILE_ENV: Final = "ML_WORKER_PROFILE"


class ProfileError(RuntimeError):
    pass


class ProfileVerifyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    profile: str
    stage: str
    reason: str


DeviceVerifier: TypeAlias = Callable[[], VerifyResult]
DecodeProbe: TypeAlias = Callable[[str], VerifyResult]


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: str
    device: DevicePolicy
    decode: DecodePolicy
    encode: EncodePolicy


PROFILE_REGISTRY: Final[Mapping[str, ProfileSpec]] = MappingProxyType(
    {
        "cuda": ProfileSpec("cuda", "cuda", "nvdec", "h264_nvenc"),
        "mps": ProfileSpec("mps", "mps", "opencv", "libx264"),
        "cpu": ProfileSpec("cpu", "cpu", "opencv", "libx264"),
    }
)


@dataclass(frozen=True, slots=True)
class BootDependencies:
    verifiers: Mapping[str, DeviceVerifier]


def _verify_cuda(source: CudaProbeSource | None) -> VerifyResult:
    result = probe_cuda(source)
    return VerifyResult(result.available, "cuda", "device", result.reason)


def _verify_mps(source: MpsProbeSource | None) -> VerifyResult:
    if source is None:
        return VerifyResult(False, "mps", "device", "MPS capability probe is not configured")
    available = source()
    reason = "MPS is available" if available else "MPS is unavailable"
    return VerifyResult(available, "mps", "device", reason)


def _verify_cpu() -> VerifyResult:
    return VerifyResult(True, "cpu", "device", "CPU is available")


def default_verifiers(
    *,
    cuda_source: CudaProbeSource | None = None,
    mps_source: MpsProbeSource | None = None,
) -> Mapping[str, DeviceVerifier]:
    """Build fail-closed device verifiers from injected capability probes."""
    return MappingProxyType(
        {
            "cuda": lambda: _verify_cuda(cuda_source),
            "mps": lambda: _verify_mps(mps_source),
            "cpu": _verify_cpu,
        }
    )


def default_decode_probe(
    decode: str,
    probes: Mapping[str, Callable[[], VerifyResult]] | None = None,
) -> VerifyResult:
    """Run an injected decode probe, failing closed when one is unavailable."""
    if probes is None or decode not in probes:
        return VerifyResult(
            False,
            "cuda" if decode == "nvdec" else "",
            "decode",
            f"{decode} capability probe is not configured",
        )
    return probes[decode]()


__all__ = [
    "ML_WORKER_PROFILE_ENV",
    "PROFILE_REGISTRY",
    "BootDependencies",
    "DecodePolicy",
    "DecodeProbe",
    "DevicePolicy",
    "DeviceVerifier",
    "EncodePolicy",
    "ProfileError",
    "ProfileSpec",
    "ProfileVerifyError",
    "VerifyResult",
    "default_decode_probe",
    "default_verifiers",
]
