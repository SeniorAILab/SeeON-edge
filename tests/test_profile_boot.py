from __future__ import annotations

import pytest

import edge.runtime.profile.registry as profile_registry
from edge.runners.device import CudaProbe
from edge.runtime.pipeline_bootstrap import BOOTSTRAP_EXIT_CODE, bootstrap_or_exit
from edge.runtime.profile.boot import (
    profile_verify_stage,
    resolve_boot_context,
    resolve_profile,
)
from edge.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
    ProfileError,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
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


def test_profile_verify_stage_missing_profile_exits() -> None:
    codes: list[int] = []

    def exit_spy(code: int) -> None:
        codes.append(code)
        raise SystemExit(code)

    stage = profile_verify_stage({})
    with pytest.raises(SystemExit):
        bootstrap_or_exit([stage], exit_fn=exit_spy)

    assert codes == [BOOTSTRAP_EXIT_CODE]


def _cuda_available() -> CudaProbe:
    return CudaProbe(available=True, reason="cuda available")


def test_nvdec_decode_probe_accepts_cuda_hwaccel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_registry, "probe_cuda", _cuda_available)
    monkeypatch.setattr(
        profile_registry, "_run_ffmpeg", lambda args: "Hardware acceleration methods:\ncuda"
    )

    result = default_decode_probe("nvdec")

    assert result.ok
    assert result.stage == "decode"


def test_nvdec_decode_probe_rejects_missing_cuvid_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profile_registry, "probe_cuda", _cuda_available)
    monkeypatch.setattr(
        profile_registry,
        "_run_ffmpeg",
        lambda args: "Hardware acceleration methods:\nvaapi\nvideotoolbox",
    )

    result = default_decode_probe("nvdec")

    assert not result.ok
    assert result.stage == "decode"
    assert "no CUDA/CUVID/NVDEC hwaccel" in result.reason


def test_nvdec_decode_probe_rejects_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_registry, "probe_cuda", _cuda_available)

    def missing_ffmpeg(args: tuple[str, ...]) -> str:
        raise FileNotFoundError

    monkeypatch.setattr(profile_registry, "_run_ffmpeg", missing_ffmpeg)

    result = default_decode_probe("nvdec")

    assert not result.ok
    assert result.reason == "ffmpeg missing"


def test_cuda_profile_fails_when_ffmpeg_has_no_nvdec_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profile_registry, "probe_cuda", _cuda_available)
    monkeypatch.setattr(
        profile_registry, "_run_ffmpeg", lambda args: "Hardware acceleration methods:\nvaapi"
    )

    with pytest.raises(ProfileVerifyError, match="no CUDA/CUVID/NVDEC hwaccel"):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"})


def test_opencv_decode_probe_ok() -> None:
    assert default_decode_probe("opencv").ok
