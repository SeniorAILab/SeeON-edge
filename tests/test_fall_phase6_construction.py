from __future__ import annotations

import pytest

from contracts.model_reference import parse_model_reference
from worker.runtime.config import local_env
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import WorkerModelsConfig

REFERENCE = "owner/model@" + "a" * 40


def test_fall_model_refuses_person_boxes() -> None:
    with pytest.raises(ValueError, match="requires box_source=pose"):
        WorkerModelsConfig(box_source="person", fall_model=parse_model_reference(REFERENCE))


def test_set_fall_model_env_selects_that_reference() -> None:
    config = local_env.worker_models_config_from_environment({"ML_WORKER_FALL_MODEL": REFERENCE})

    assert config.fall_model == parse_model_reference(REFERENCE)
    assert config.fall is None


def test_mutable_ref_is_a_config_error() -> None:
    with pytest.raises(WorkerConfigError, match="branch and tag names are not allowed"):
        local_env.worker_models_config_from_environment(
            {"ML_WORKER_FALL_MODEL": "owner/model@main"}
        )
