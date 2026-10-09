from __future__ import annotations

from pathlib import Path

import pytest

from tests_support.pose_bbox56_bundle_artifact import write_admitted_pose_bbox56_bundle
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.runtime.config import WorkerConfig
from worker.runtime.config.local_env import worker_models_config_from_environment
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime


def _runtime(env: dict[str, str], state_dir: Path) -> WorkerRuntime:
    config = WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    ).model_copy(update={"models": worker_models_config_from_environment(env)})
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=object(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        state_dir=state_dir,
    )


def test_bundle_dir_env_yields_a_runner_for_a_valid_bundle(tmp_path: Path) -> None:
    bundle = write_admitted_pose_bbox56_bundle(tmp_path / "src")
    runtime = _runtime({"ML_WORKER_FALL_BUNDLE_DIR": str(bundle)}, tmp_path)

    runner = runtime._create_fall_model()

    assert isinstance(runner, OrtPoseBbox56Runner)
    assert runner.device == "cpu"
    assert runner.artifact_digest == runtime._loaded_fall_bundle.published_weights_digest  # type: ignore[union-attr]


def test_bundle_dir_env_refuses_a_tampered_bundle_naming_dir_and_cause(tmp_path: Path) -> None:
    bundle = write_admitted_pose_bbox56_bundle(tmp_path / "src")
    (bundle / "model.onnx").write_bytes(b"swapped")
    runtime = _runtime({"ML_WORKER_FALL_BUNDLE_DIR": str(bundle)}, tmp_path)

    with pytest.raises(ModelLoadError) as raised:
        runtime._create_fall_model()

    assert str(raised.value) == (
        f"fall bundle {bundle}: member model.onnx does not match its declared hash"
    )
