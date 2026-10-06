from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime import bootstrap as boot
from worker.runtime.lease import GpuLeaseUnavailableError
from worker.runtime.profile.registry import BootDependencies, VerifyResult


def test_run_stages_runs_in_order_and_collects_outputs() -> None:
    calls: list[str] = []
    stages = (
        boot.Stage("gpu_verify", lambda: (calls.append("gpu"), "cuda")[1]),
        boot.Stage("backend_init", lambda: (calls.append("backend"), "bundle")[1]),
        boot.Stage("warmup", lambda: (calls.append("warmup"), "ready")[1]),
    )
    result = boot.run_stages(stages)
    assert calls == ["gpu", "backend", "warmup"]
    assert result.outputs == {"gpu_verify": "cuda", "backend_init": "bundle", "warmup": "ready"}


def test_stage_failure_raises_bootstrap_stage_error_naming_the_stage() -> None:
    def boom() -> None:
        raise RuntimeError("backend down")

    stages = (boot.Stage("gpu_verify", lambda: "cuda"), boot.Stage("backend_init", boom))
    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages(stages)
    assert exc.value.stage == "backend_init"
    assert "backend_init" in str(exc.value)
    assert "backend down" in str(exc.value)


def test_bootstrap_or_exit_exits_with_stage_exit_code_on_failure() -> None:
    exit_codes: list[int] = []

    def boom() -> None:
        raise RuntimeError("no compiled arch kernels (sm_120 missing); see ADR-0002")

    stages = (boot.Stage("gpu_verify", boom),)
    with pytest.raises(boot.BootstrapStageError):
        boot.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == [boot.GENERIC_RUNTIME_EXIT_CODE]
    assert boot.GENERIC_RUNTIME_EXIT_CODE != 0


def test_bootstrap_or_exit_returns_result_when_all_pass() -> None:
    exit_codes: list[int] = []
    stages = (boot.Stage("gpu_verify", lambda: "cuda"), boot.Stage("warmup", lambda: "ready"))
    result = boot.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == []
    assert result.outputs["gpu_verify"] == "cuda"


def test_gpu_lease_stage_fails_fast_when_lease_unavailable(tmp_path: Path) -> None:
    context = boot.BootstrapContext()

    def refuse() -> boot.GpuLease:
        raise GpuLeaseUnavailableError(tmp_path / ".gpu.lease")

    stage = boot.gpu_lease_stage(context, acquire=refuse)
    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages((stage,))
    assert exc.value.exit_code == boot.REFUSE_TO_START_EXIT_CODE
    assert context.lease is None


def test_profile_device_stage_fails_fast_and_publishes_no_profile_on_failure() -> None:
    context = boot.BootstrapContext()
    deps = BootDependencies({"flow": lambda: VerifyResult(True, "flow", "device", "available")})
    stage = boot.profile_device_stage(context, {"ML_WORKER_PROFILE": "nvidia"}, deps)

    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages((stage,))
    assert exc.value.exit_code == boot.REFUSE_TO_START_EXIT_CODE
    assert "ADR-0002: unsupported ML_WORKER_PROFILE 'nvidia'; set flow" in str(exc.value)
    assert context.profile is None


def test_per_camera_ordinary_failure_degrades_only_that_camera() -> None:
    def boom() -> None:
        raise RuntimeError("nvdec spawn failed")

    ok = boot.run_camera_stage("camA", lambda: None)
    degraded = boot.run_camera_stage("camB", boom)
    assert ok.ok is True and ok.reason is None
    assert degraded.ok is False and "nvdec" in degraded.reason
    assert degraded.camera_id == "camB"


def test_per_camera_fatal_accelerator_error_is_reraised_not_degraded() -> None:
    def boom() -> None:
        raise FatalAcceleratorError("CUDA error: device-side assert triggered", camera_id="camC")

    with pytest.raises(FatalAcceleratorError):
        boot.run_camera_stage("camC", boom)
