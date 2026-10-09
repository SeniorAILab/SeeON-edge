from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from contracts.model import POSE_BBOX56_PREPROCESSING_IDENTITY
from contracts.model_reference import parse_model_reference
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    FallModelConfig,
    WorkerConfig,
    WorkerModelsConfig,
)

ML_WORKER_CLIP_RECORDING_ENABLED_ENV: Final = "ML_WORKER_CLIP_RECORDING_ENABLED"
WORKER_REPLAY_TRACE_DIR_ENV: Final = "WORKER_REPLAY_TRACE_DIR"
ML_WORKER_FALL_MODEL_ENV: Final = "ML_WORKER_FALL_MODEL"

_RETIRED_WORKER_ENV: Final = frozenset(
    {
        ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
        "CLIP_STORE_DIR",
        "EDGE_CAMERA_CONFIG",
        "EDGE_CAMERA_CONFIG_FILE",
        "ML_WORKER_DEV_MJPEG",
        "ML_WORKER_DEV_MJPEG_HOST",
        "ML_WORKER_DEV_MJPEG_PORT",
        "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED",
        "RELAY_URL",
    }
)


def reject_retired_worker_environment(environ: Mapping[str, str]) -> None:
    present = sorted(_RETIRED_WORKER_ENV.intersection(environ))
    if present:
        raise WorkerConfigError(
            "retired edge environment key(s): "
            + ", ".join(present)
            + "; use the versioned worker config authority"
        )


_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})
_DEFAULT_TYPE: Final = "pose-bbox56-proxy-v0"
_DEFAULT_WEIGHTS: Final = "model.pt"
_DEFAULT_ARCHITECTURE: Final = "arch.json"
_DEFAULT_ARTIFACT_DIR: Final = "models/fall/pose-bbox56-gru"
_DEFAULT_WINDOW: Final = 30
_DEFAULT_STRIDE: Final = 5
_DEFAULT_OPERATING_THRESHOLD: Final = 0.5
_DEFAULT_SCHEMA_VERSION: Final = 2
_DEFAULT_PREPROCESSING_IDENTITY: Final = POSE_BBOX56_PREPROCESSING_IDENTITY
_FETCH_MODELS_HINT: Final = (
    "run scripts/fetch-models.sh to download the packaged pose+bbox56 model weights"
)


def _bool_env(name: str, env: Mapping[str, str]) -> bool | None:
    raw = env.get(name, "").strip().lower()
    if raw == "":
        return None
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise WorkerConfigError(f"{name} must be a boolean ({sorted(_TRUTHY | _FALSY)}), got {raw!r}")


def clip_recording_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ClipRecordingConfig:
    env = os.environ if environ is None else environ
    explicit = _bool_env(ML_WORKER_CLIP_RECORDING_ENABLED_ENV, env)
    return ClipRecordingConfig() if explicit is None else ClipRecordingConfig(enabled=explicit)


def fall_model_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> FallModelConfig:
    env = os.environ if environ is None else environ
    framework: Literal["pytorch", "onnxruntime"] = (
        "onnxruntime" if env.get("ML_WORKER_PROFILE", "").strip() == "flow" else "pytorch"
    )
    try:
        return FallModelConfig(
            type=_DEFAULT_TYPE,
            framework=framework,
            mode="sequence",
            artifact_dir=Path(_DEFAULT_ARTIFACT_DIR),
            weights=_DEFAULT_WEIGHTS,
            architecture=_DEFAULT_ARCHITECTURE,
            window=_DEFAULT_WINDOW,
            stride=_DEFAULT_STRIDE,
            input_shape=(_DEFAULT_WINDOW, 56),
            operating_threshold=_DEFAULT_OPERATING_THRESHOLD,
            schema_version=_DEFAULT_SCHEMA_VERSION,
            preprocessing_identity=_DEFAULT_PREPROCESSING_IDENTITY,
        )
    except ValidationError as error:
        raise WorkerConfigError(
            "packaged default pose+bbox56 fall model is not fully provisioned at "
            f"{_DEFAULT_ARTIFACT_DIR!r} ({error}); {_FETCH_MODELS_HINT}"
        ) from error


def worker_models_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> WorkerModelsConfig:
    env = os.environ if environ is None else environ
    reference = env.get(ML_WORKER_FALL_MODEL_ENV, "").strip()
    if reference:
        try:
            return WorkerModelsConfig(fall_model=parse_model_reference(reference))
        except ValueError as error:
            raise WorkerConfigError(f"{ML_WORKER_FALL_MODEL_ENV}: {error}") from error
    return WorkerModelsConfig(
        fall=fall_model_config_from_environment(environ),
    )


def resolve_local_overrides(
    yaml_config: WorkerConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[WorkerModelsConfig, ClipRecordingConfig, DevMjpegConfig | None]:
    env = os.environ if environ is None else environ
    environment_models = worker_models_config_from_environment(env)
    yaml_models = yaml_config.models if yaml_config is not None else None
    clip = (
        yaml_config.clip
        if yaml_config is not None and yaml_config.clip.enabled
        else clip_recording_config_from_environment(env)
    )
    dev_mjpeg = (
        yaml_config.dev_mjpeg if yaml_config is not None and yaml_config.dev_mjpeg.enabled else None
    )
    if environment_models.fall_model is not None:
        if yaml_models is not None:
            raise WorkerConfigError(
                f"{ML_WORKER_FALL_MODEL_ENV} cannot coexist with a packaged fall model"
            )
        return environment_models, clip, dev_mjpeg
    models = WorkerModelsConfig(
        fall=(
            yaml_models.fall
            if yaml_models is not None and yaml_models.fall is not None
            else environment_models.fall
        ),
        box_source=(
            yaml_models.box_source
            if yaml_models is not None and yaml_models.fall is not None
            else environment_models.box_source
        ),
    )
    return models, clip, dev_mjpeg


def replay_trace_directory_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    env = os.environ if environ is None else environ
    raw = env.get(WORKER_REPLAY_TRACE_DIR_ENV, "").strip()
    return None if not raw else Path(raw)


__all__ = [
    "ML_WORKER_CLIP_RECORDING_ENABLED_ENV",
    "ML_WORKER_FALL_MODEL_ENV",
    "fall_model_config_from_environment",
    "reject_retired_worker_environment",
    "replay_trace_directory_from_environment",
    "resolve_local_overrides",
    "worker_models_config_from_environment",
]
