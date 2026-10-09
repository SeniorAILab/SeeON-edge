from __future__ import annotations

from pathlib import Path

import pytest

from worker.runtime.config import local_env
from worker.runtime.config.worker_models import WorkerModelsConfig


def test_bundle_dir_refuses_person_boxes() -> None:
    with pytest.raises(ValueError, match="requires box_source=pose"):
        WorkerModelsConfig(box_source="person", fall_bundle_dir=Path("/b"))


def test_set_bundle_dir_env_selects_that_directory(tmp_path: Path) -> None:
    config = local_env.worker_models_config_from_environment(
        {"ML_WORKER_FALL_BUNDLE_DIR": str(tmp_path)}
    )

    assert config.fall_bundle_dir == tmp_path.resolve()
    assert config.fall is None


def test_unset_bundle_dir_env_keeps_the_packaged_default(packaged_fall_bundle: Path) -> None:
    for env in ({}, {"ML_WORKER_FALL_BUNDLE_DIR": ""}):
        config = local_env.worker_models_config_from_environment(env)
        assert config.fall_bundle_dir is None
        assert config.fall is not None
        assert config.fall.artifact_dir == packaged_fall_bundle.resolve()
