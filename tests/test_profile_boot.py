from __future__ import annotations

import pytest

from worker.runtime.profile.boot import (
    resolve_boot_context,
    resolve_profile,
)
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
    ProfileError,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
    default_verifiers,
)


def _result(profile: str, *, ok: bool = True, stage: str = "device") -> VerifyResult:
    return VerifyResult(ok=ok, profile=profile, stage=stage, reason="test result")


def _deps(profile: str, *, ok: bool = True) -> BootDependencies:
    return BootDependencies({profile: lambda: _result(profile, ok=ok)})


def _decode_ok(decode: str) -> VerifyResult:
    return VerifyResult(True, "", "decode", f"{decode} available")


@pytest.mark.parametrize("env", [{}, {ML_WORKER_PROFILE_ENV: "  "}])
def test_resolve_profile_missing_env_raises(env: dict[str, str]) -> None:
    with pytest.raises(ProfileError):
        resolve_profile(env)


def test_resolve_profile_unknown_raises() -> None:
    with pytest.raises(ProfileError):
        resolve_profile({ML_WORKER_PROFILE_ENV: "tpu"})


def test_cuda_profile_verify_true() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda"), _decode_ok
    )

    assert context.device == "cuda"
    assert context.decode == "nvdec"


def test_cuda_profile_verify_false() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda", ok=False), _decode_ok)


def test_mps_profile_verify_true() -> None:
    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "mps"}, _deps("mps"), _decode_ok)

    assert context.device == "mps"
    assert context.decode == "opencv"


def test_mps_profile_verify_false() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "mps"}, _deps("mps", ok=False), _decode_ok)


def test_cpu_profile_ok() -> None:
    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "cpu"}, _deps("cpu"), _decode_ok)

    assert context.device == "cpu"
    assert context.decode == "opencv"


def test_verifier_exception_becomes_profile_verify_error() -> None:
    def verifier() -> VerifyResult:
        raise RuntimeError("unavailable")

    with pytest.raises(ProfileVerifyError):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cpu"}, BootDependencies({"cpu": verifier}), _decode_ok
        )


def test_decode_preflight_incompat_raises() -> None:
    def decode_probe(decode: str) -> VerifyResult:
        assert decode == "nvdec"
        return VerifyResult(False, "cuda", "decode", "NVDEC unavailable")

    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda"), decode_probe)


def test_legacy_decode_conflict_rejected() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cpu", "ML_RTSP_BACKEND": "nvdec"},
            _deps("cpu"),
            _decode_ok,
        )


def test_legacy_matching_allowed() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cpu", "ML_RTSP_BACKEND": "opencv"},
        _deps("cpu"),
        _decode_ok,
    )

    assert context.decode == "opencv"


def test_profile_registry_exact_keys() -> None:
    assert set(PROFILE_REGISTRY) == {"cuda", "mps", "cpu"}
    assert PROFILE_REGISTRY["cuda"].device == "cuda"
    assert PROFILE_REGISTRY["cuda"].decode == "nvdec"
    assert PROFILE_REGISTRY["mps"].device == "mps"
    assert PROFILE_REGISTRY["mps"].decode == "opencv"
    assert PROFILE_REGISTRY["cpu"].device == "cpu"
    assert PROFILE_REGISTRY["cpu"].decode == "opencv"


def _cuda_available() -> CudaProbe:
    return CudaProbe(available=True, reason="cuda available")


def test_cuda_device_verify_uses_injected_probe_source() -> None:
    deps = BootDependencies(default_verifiers(cuda_source=_cuda_available))

    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, deps, _decode_ok)

    assert context.device == "cuda"


def test_default_decode_probe_fails_closed_without_injected_probes() -> None:
    result = default_decode_probe("nvdec")

    assert not result.ok
    assert result.stage == "decode"
    assert result.profile == "cuda"
    assert result.reason == "nvdec capability probe is not configured"


def test_default_decode_probe_fails_closed_for_opencv_without_injected_probes() -> None:
    result = default_decode_probe("opencv")

    assert not result.ok
    assert result.profile == ""
    assert result.reason == "opencv capability probe is not configured"


def test_default_decode_probe_uses_injected_probe_when_configured() -> None:
    probes = {"nvdec": lambda: VerifyResult(True, "cuda", "decode", "nvdec available")}

    result = default_decode_probe("nvdec", probes)

    assert result.ok
    assert result.reason == "nvdec available"


def test_cuda_profile_fails_closed_when_no_decode_probe_configured() -> None:
    deps = BootDependencies(default_verifiers(cuda_source=_cuda_available))

    with pytest.raises(ProfileVerifyError, match="nvdec capability probe is not configured"):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, deps)
