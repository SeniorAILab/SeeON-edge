from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.local_env import (
    fall_model_config_from_environment,
    reject_retired_worker_environment,
    worker_models_config_from_environment,
)
from worker.runtime.config.worker_models import WorkerModelsConfig

_PACKAGED_DEFAULT_ARTIFACT_DIR = Path("models/fall/pose-bbox56-gru")
_PREPROCESSING_IDENTITY = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"


def test_default_env_resolves_packaged_pose_bbox56_config(packaged_fall_bundle: Path) -> None:
    config = fall_model_config_from_environment({})

    assert config.type == "pose-bbox56-proxy-v0"
    assert config.artifact_dir == packaged_fall_bundle.resolve()
    assert config.window == 30
    assert config.stride == 5
    assert config.input_shape == (30, 56)
    assert config.schema_version == 2
    assert config.operating_threshold == 0.5
    assert config.preprocessing_identity == _PREPROCESSING_IDENTITY


def test_bundle_runner_loads_and_predicts_from_packaged_default(
    packaged_fall_bundle: Path,
) -> None:
    from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner

    runner = PoseBbox56BundleRunner.from_artifact_dir(packaged_fall_bundle)
    probabilities = runner.predict(np.zeros((30, 56), dtype=np.float32))

    assert runner.device == "cpu"
    assert 0.0 <= probabilities.fall_transition <= 1.0
    assert probabilities.fallen == 0.0


def test_default_env_missing_weights_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(WorkerConfigError, match="missing model.pt"):
        worker_models_config_from_environment({})


def test_models_config_refuses_when_no_fall_model_is_available() -> None:
    with pytest.raises(ValueError, match="no fall model configured"):
        WorkerModelsConfig()


def _write_fake_packaged_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_pose_bbox56_bundle(tmp_path / _PACKAGED_DEFAULT_ARTIFACT_DIR)


def test_default_env_with_no_overrides_resolves_packaged_manifest_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fake_packaged_default(tmp_path, monkeypatch)

    config = fall_model_config_from_environment({})

    assert config.window == 30
    assert config.stride == 5
    assert config.operating_threshold == 0.5


def test_model_policy_environment_keys_are_retired_explicitly() -> None:
    environ = {
        "ML_WORKER_FALL_MODEL_WINDOW": "45",
        "ML_WORKER_FALL_MODEL_STRIDE": "9",
        "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD": "0.5",
    }

    with pytest.raises(WorkerConfigError) as excinfo:
        reject_retired_worker_environment(environ)

    assert all(name in str(excinfo.value) for name in environ)
    assert "versioned worker config authority" in str(excinfo.value)


def test_retired_manifest_environment_keys_fail_instead_of_warning() -> None:
    environ = {
        "ML_WORKER_FALL_MODEL_SCHEMA_VERSION": "2",
        "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY": "some-other-identity",
    }

    with pytest.raises(WorkerConfigError) as excinfo:
        reject_retired_worker_environment(environ)

    assert all(name in str(excinfo.value) for name in environ)
