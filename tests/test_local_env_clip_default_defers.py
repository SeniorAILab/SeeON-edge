from __future__ import annotations

import pytest

from worker.runtime.config.local_env import (
    ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
    clip_recording_config_from_environment,
)
from worker.runtime.config.worker_models import ClipRecordingConfig


def test_silent_env_defers_to_clip_recording_config_model_default() -> None:
    config = clip_recording_config_from_environment({})

    assert config.enabled == ClipRecordingConfig().enabled


@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False)])
def test_explicit_env_value_wins_outright_over_model_default(raw: str, expected: bool) -> None:
    config = clip_recording_config_from_environment({ML_WORKER_CLIP_RECORDING_ENABLED_ENV: raw})

    assert config.enabled is expected
