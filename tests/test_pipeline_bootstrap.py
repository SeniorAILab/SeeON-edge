"""6-stage 2-tier bootstrap gate injection (ADR-0002, acceptance 4).

Global-stage failure -> process exits non-zero (no CPU fallback);
per-camera-stage failure -> that camera degrades only, no exit.
"""

from __future__ import annotations

import pytest

from edge.runners.device import GpuUnavailableError
from edge.runtime import pipeline_bootstrap as pb


def test_global_bootstrap_runs_stages_in_order_and_collects_outputs():
    calls = []
    stages = [
        pb.Stage("gpu_verify", lambda: (calls.append("gpu"), "cuda")[1]),
        pb.Stage("backend_init", lambda: (calls.append("backend"), "bundle")[1]),
        pb.Stage("warmup", lambda: (calls.append("warmup"), "ready")[1]),
    ]
    result = pb.run_global_bootstrap(stages)
    assert calls == ["gpu", "backend", "warmup"]
    assert result.outputs == {"gpu_verify": "cuda", "backend_init": "bundle", "warmup": "ready"}


def test_global_stage_failure_raises_and_names_stage():
    def boom():
        raise RuntimeError("backend down")

    stages = [pb.Stage("gpu_verify", lambda: "cuda"), pb.Stage("backend_init", boom)]
    with pytest.raises(pb.GlobalBootstrapError) as exc:
        pb.run_global_bootstrap(stages)
    assert "backend_init" in str(exc.value)
    assert "backend down" in str(exc.value)


def test_bootstrap_or_exit_exits_nonzero_on_global_failure():
    exit_codes = []

    def failing_gpu():
        raise GpuUnavailableError("no compiled arch kernels (sm_120 missing); see ADR-0002")

    stages = [pb.Stage("gpu_verify", failing_gpu)]
    with pytest.raises(pb.GlobalBootstrapError):
        pb.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == [pb.BOOTSTRAP_EXIT_CODE]
    assert pb.BOOTSTRAP_EXIT_CODE != 0


def test_bootstrap_or_exit_returns_result_when_all_pass():
    exit_codes = []
    stages = [pb.Stage("gpu_verify", lambda: "cuda"), pb.Stage("warmup", lambda: "ready")]
    result = pb.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == []
    assert result.outputs["gpu_verify"] == "cuda"


def test_gpu_verify_stage_fails_fast_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(
        "edge.runtime.pipeline_bootstrap.require_gpu_device",
        lambda explicit=None: (_ for _ in ()).throw(GpuUnavailableError("unusable")),
    )
    with pytest.raises(pb.GlobalBootstrapError):
        pb.run_global_bootstrap([pb.gpu_verify_stage()])


def test_gpu_verify_stage_honors_explicit_cpu_optin(monkeypatch):
    monkeypatch.setattr(
        "edge.runtime.pipeline_bootstrap.require_gpu_device",
        lambda explicit=None: explicit if explicit is not None else "cuda",
    )
    result = pb.run_global_bootstrap([pb.gpu_verify_stage("cpu")])
    assert result.outputs["gpu_verify"] == "cpu"


def test_per_camera_failure_degrades_only_that_camera():
    def boom():
        raise RuntimeError("nvdec spawn failed")

    ok = pb.run_camera_stage("camA", lambda: None)
    degraded = pb.run_camera_stage("camB", boom)
    assert ok.ok is True and ok.reason is None
    assert degraded.ok is False and "nvdec" in degraded.reason
    assert degraded.camera_id == "camB"
