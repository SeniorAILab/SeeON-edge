from __future__ import annotations

import pytest

from worker.runtime.profile.boot import BootContext, reject_legacy_conflicts
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    ProfileVerifyError,
    default_verifiers,
)


def test_ml_worker_profile_env_name() -> None:
    assert ML_WORKER_PROFILE_ENV == "ML_WORKER_PROFILE"


def test_default_verifiers_cpu_always_available() -> None:
    result = default_verifiers()["cpu"]()

    assert result.ok
    assert result.profile == "cpu"
    assert result.stage == "device"
    assert result.reason == "CPU is available"


def test_default_verifiers_cuda_without_source_fails_closed() -> None:
    result = default_verifiers()["cuda"]()

    assert not result.ok
    assert result.profile == "cuda"
    assert result.stage == "device"
    assert result.reason == "CUDA capability probe is not configured"


def test_default_verifiers_cuda_forwards_injected_probe_reason() -> None:
    verifiers = default_verifiers(
        cuda_source=lambda: CudaProbe(available=True, reason="2 devices detected")
    )

    result = verifiers["cuda"]()

    assert result.ok
    assert result.reason == "2 devices detected"


def test_default_verifiers_mps_without_source_fails_closed() -> None:
    result = default_verifiers()["mps"]()

    assert not result.ok
    assert result.profile == "mps"
    assert result.reason == "MPS capability probe is not configured"


def test_default_verifiers_mps_forwards_injected_source() -> None:
    available = default_verifiers(mps_source=lambda: True)["mps"]()
    assert available.ok
    assert available.reason == "MPS is available"

    unavailable = default_verifiers(mps_source=lambda: False)["mps"]()
    assert not unavailable.ok
    assert unavailable.reason == "MPS is unavailable"


@pytest.mark.parametrize(
    "legacy_env",
    [
        {"ML_RTSP_BACKEND": "nvdec"},
        {"ML_DEFAULT_DECODE_BACKEND": "nvdec"},
    ],
)
def test_reject_legacy_conflicts_raises_for_incompatible_backend(
    legacy_env: dict[str, str],
) -> None:
    with pytest.raises(ProfileVerifyError):
        reject_legacy_conflicts(PROFILE_REGISTRY["cpu"], legacy_env)


def test_reject_legacy_conflicts_allows_auto() -> None:
    assert reject_legacy_conflicts(PROFILE_REGISTRY["cpu"], {"ML_RTSP_BACKEND": "auto"}) is None


def test_reject_legacy_conflicts_allows_matching_backend() -> None:
    result = reject_legacy_conflicts(PROFILE_REGISTRY["cuda"], {"ML_RTSP_BACKEND": "nvdec"})
    assert result is None


def test_boot_context_carries_resolved_profile_fields() -> None:
    spec = PROFILE_REGISTRY["cuda"]

    context = BootContext(profile=spec, device=spec.device, decode=spec.decode, encode=spec.encode)

    assert context.profile is spec
    assert context.device == "cuda"
    assert context.decode == "nvdec"
    assert context.encode == "h264_nvenc"
