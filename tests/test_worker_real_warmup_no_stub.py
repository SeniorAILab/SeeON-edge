from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.runtime.config import WorkerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

_TASK_ARTIFACTS: dict[str, Path] = {
    "pose": REPO_ROOT / "models" / "pose" / "yolo26n-pose.pt",
    "person": REPO_ROOT / "models" / "person" / "yolo26n.pt",
    "bed": REPO_ROOT / "models" / "bed" / "yolo26l-seg.pt",
}


def _require(task: str) -> Path:
    artifact = _TASK_ARTIFACTS[task]
    if not artifact.is_file():
        pytest.skip(f"{task} weights are not present at {artifact}")
    return artifact


_REAL_WARMUP_TIMEOUT_SECONDS = 60.0
_REAL_WARMUP_COMPLETED = "REAL_WARMUP_COMPLETED"
_REAL_WARMUP_SCRIPT = f"""
import sys

from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import default_registry

TASK = sys.argv[1]
serving = InProcessServingClient(registry=default_registry())
adapter = serving.create(TASK, device="cpu")
adapter.warmup()
print("{_REAL_WARMUP_COMPLETED}:" + TASK, flush=True)
"""


@pytest.mark.heavy
@pytest.mark.parametrize("task", sorted(_TASK_ARTIFACTS))
def test_real_warmup_runs_a_genuine_forward_through_the_production_serving_path(
    task: str,
) -> None:
    _ = _require(task)

    completed = subprocess.run(
        [sys.executable, "-c", _REAL_WARMUP_SCRIPT, task],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=_REAL_WARMUP_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"{_REAL_WARMUP_COMPLETED}:{task}" in completed.stdout


def _example_fall_config() -> dict[str, object]:
    example = yaml.safe_load(
        (REPO_ROOT / "worker" / "ml-worker.example.yaml").read_text(encoding="utf-8")
    )
    return dict(example["models"]["fall"])


def test_example_config_fall_contract_matches_the_packaged_bundle() -> None:
    fall_cfg = _example_fall_config()
    assert fall_cfg["type"] == "pose-bbox56-proxy-v0"
    assert fall_cfg["input_shape"] == [30, 56]
    assert (fall_cfg["window"], fall_cfg["stride"]) == (30, 5)
    assert fall_cfg["operating_threshold"] == 0.5
    assert fall_cfg["schema_version"] == 2
    assert fall_cfg["preprocessing_identity"] == "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"

    artifact_dir = REPO_ROOT / "models" / "fall" / "pose-bbox56-gru"
    if not (artifact_dir / "bundle-manifest.json").is_file():
        pytest.skip(f"packaged fall bundle is not present at {artifact_dir}")
    runner = PoseBbox56BundleRunner.from_artifact_dir(artifact_dir, device="cpu")
    assert runner.device == "cpu"


def test_example_config_fall_contract_boots_against_a_synthesized_bundle(
    tmp_path: Path,
) -> None:
    fall_cfg = _example_fall_config()
    fall_cfg["artifact_dir"] = str(write_pose_bbox56_bundle(tmp_path / "pose-bbox56-gru"))
    config = WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [],
            "models": {"fall": fall_cfg},
        }
    )
    assert config.models.fall is not None
    assert config.models.fall.input_shape == (30, 56)

    runner = PoseBbox56BundleRunner.from_artifact_dir(config.models.fall.artifact_dir)
    runner.warmup()
    assert runner.device == "cpu"


def test_fall_classifier_is_constructed_and_warmed_on_the_cpu_before_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import worker.runtime.worker as worker_module
    from worker.domains import DETECTION_MODULE_REGISTRY
    from worker.runtime.flow.media_plane import FlowMediaPlane
    from worker.runtime.lease import GpuLease
    from worker.runtime.profile.boot import BootContext
    from worker.runtime.profile.registry import PROFILE_REGISTRY

    def _binding(task: str) -> object:
        component_id = "fall-classifier" if task == "fall" else task
        return next(
            binding
            for definition in DETECTION_MODULE_REGISTRY.definitions
            for binding in definition.shared_bindings
            if binding.component_id == component_id
        )

    class _Runner:
        device = "cpu"

        def __init__(self, task: str) -> None:
            binding = _binding(task)
            self.artifact_digest = binding.artifact_digest
            self.preprocessing_identity = binding.preprocessing_identity
            self.warmup_calls = 0

        def __call__(self, _image: object) -> object:
            raise AssertionError("warmup must not run camera inference here")

        def predict(self, _features: object) -> object:
            raise AssertionError("warmup must not score a window here")

        def warmup(self) -> None:
            self.warmup_calls += 1

    class _Serving:
        def create(self, task: str, **_options: object) -> object:
            return _Runner(task)

    fall_runner = _Runner("fall")

    def _create_fall_model(self: Any) -> object:
        self._loaded_fall_bundle = SimpleNamespace(
            runner=fall_runner,
            published_weights_digest="fall-digest",
            preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        )
        return fall_runner

    monkeypatch.setattr(worker_module.WorkerRuntime, "_create_fall_model", _create_fall_model)
    monkeypatch.setattr(
        worker_module, "verify_flow_boot_inputs", lambda _env, **_kwargs: {"engine": "verified"}
    )

    class _FlowMediaPlane:
        def bind_live_frames(self, _frames: object) -> None:
            pass

    config = WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [],
            "models": {
                "fall": {
                    **_example_fall_config(),
                    "artifact_dir": str(write_pose_bbox56_bundle(tmp_path / "pose-bbox56-gru")),
                }
            },
        }
    )
    runtime = worker_module.WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=_Serving(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        flow_media_plane=cast(FlowMediaPlane, _FlowMediaPlane()),
    )
    profile = PROFILE_REGISTRY["flow"]
    boot = BootContext(profile, profile.device, profile.decode, profile.encode)
    _ = runtime._initialize_models(boot)
    warmed = runtime._warm_models()

    assert "fall-classifier" in warmed
    assert fall_runner.warmup_calls == 1
