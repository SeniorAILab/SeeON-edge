from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from contracts.model_reference import parse_model_reference
from tests_support.pose_bbox56_bundle_artifact import write_admitted_pose_bbox56_bundle
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.runtime.config import WorkerConfig, local_env
from worker.runtime.config.local_env import worker_models_config_from_environment
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime

REVISION = "b" * 40
REFERENCE = f"owner/model@{REVISION}"


def _land(models_root: Path, source: Path) -> Path:
    landed = parse_model_reference(REFERENCE).bundle_dir(models_root)
    landed.parent.mkdir(parents=True)
    shutil.copytree(source, landed)
    return landed


def _runtime(env: dict[str, str], state_dir: Path, models_root: Path) -> WorkerRuntime:
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
    ).model_copy(
        update={
            "models": worker_models_config_from_environment(env).model_copy(
                update={"models_root": models_root}
            )
        }
    )
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=object(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        state_dir=state_dir,
    )


def test_fall_model_env_loads_from_the_derived_dir(tmp_path: Path) -> None:
    _land(tmp_path / "models", write_admitted_pose_bbox56_bundle(tmp_path / "src"))
    runtime = _runtime({"ML_WORKER_FALL_MODEL": REFERENCE}, tmp_path, tmp_path / "models")

    runner = runtime._create_fall_model()

    assert isinstance(runner, OrtPoseBbox56Runner)
    assert runner.device == "cpu"
    assert runner.artifact_digest == runtime._loaded_fall_bundle.published_weights_digest  # type: ignore[union-attr]


def test_fall_model_env_refuses_a_tampered_bundle_naming_reference_and_cause(
    tmp_path: Path,
) -> None:
    landed = _land(tmp_path / "models", write_admitted_pose_bbox56_bundle(tmp_path / "src"))
    (landed / "model.onnx").write_bytes(b"swapped")
    runtime = _runtime({"ML_WORKER_FALL_MODEL": REFERENCE}, tmp_path, tmp_path / "models")

    with pytest.raises(ModelLoadError) as raised:
        runtime._create_fall_model()

    assert str(raised.value) == (
        f"fall model {REFERENCE}: member model.onnx does not match its declared hash"
    )


def test_fall_model_env_refuses_a_missing_dir_naming_the_reference(tmp_path: Path) -> None:
    runtime = _runtime({"ML_WORKER_FALL_MODEL": REFERENCE}, tmp_path, tmp_path / "models")

    with pytest.raises(ModelLoadError, match=f"^fall model {REFERENCE}: unreadable or malformed"):
        runtime._create_fall_model()


def test_unset_fall_model_env_keeps_the_packaged_default(packaged_fall_bundle: Path) -> None:
    for env in ({}, {"ML_WORKER_FALL_MODEL": ""}):
        config = local_env.worker_models_config_from_environment(env)
        assert config.fall_model is None
        assert config.fall is not None
        assert config.fall.artifact_dir == packaged_fall_bundle.resolve()
