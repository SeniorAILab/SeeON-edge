from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from contracts.model_selection import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    FallModelConfig,
    SelectedFallBundleConfig,
    WorkerConfig,
    WorkerModelsConfig,
)
from worker.runtime.provenance.model_bundle import (
    ModelBundleAdmissionError,
    desired_model_bundle_from_selection_document,
)

LOGGER: Final = logging.getLogger(__name__)

ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV: Final = "ML_WORKER_FALL_MODEL_ARTIFACT_DIR"
ML_WORKER_FALL_MODEL_TYPE_ENV: Final = "ML_WORKER_FALL_MODEL_TYPE"
ML_WORKER_FALL_MODEL_WEIGHTS_ENV: Final = "ML_WORKER_FALL_MODEL_WEIGHTS"
ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV: Final = "ML_WORKER_FALL_MODEL_ARCHITECTURE"
ML_WORKER_FALL_MODEL_WINDOW_ENV: Final = "ML_WORKER_FALL_MODEL_WINDOW"
ML_WORKER_FALL_MODEL_STRIDE_ENV: Final = "ML_WORKER_FALL_MODEL_STRIDE"
ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: Final = "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD"
ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV: Final = "ML_WORKER_FALL_MODEL_SCHEMA_VERSION"
ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV: Final = (
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY"
)
ML_WORKER_CLIP_RECORDING_ENABLED_ENV: Final = "ML_WORKER_CLIP_RECORDING_ENABLED"
WORKER_REPLAY_TRACE_DIR_ENV: Final = "WORKER_REPLAY_TRACE_DIR"
FALL_SELECTION_PATH: Final = Path("/app/model-selection.json")
FALL_MODELS_ROOT: Final = Path("/models")

_RETIRED_WORKER_ENV: Final = frozenset(
    {
        ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
        ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV,
        ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV,
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV,
        ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
        ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
        ML_WORKER_FALL_MODEL_STRIDE_ENV,
        ML_WORKER_FALL_MODEL_TYPE_ENV,
        ML_WORKER_FALL_MODEL_WEIGHTS_ENV,
        ML_WORKER_FALL_MODEL_WINDOW_ENV,
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
    "run scripts/fetch-models.sh to download the packaged pose+bbox56 model "
    "weights (or set ML_WORKER_FALL_MODEL_ARTIFACT_DIR to point at an "
    "already-provisioned artifact directory)"
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


def _required_str(name: str, env: Mapping[str, str], *, because: str) -> str:
    raw = env.get(name, "").strip()
    if not raw:
        raise WorkerConfigError(f"{name} is required {because}")
    return raw


def _required_int(name: str, env: Mapping[str, str], *, because: str) -> int:
    raw = _required_str(name, env, because=because)
    try:
        return int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error


def _required_float(name: str, env: Mapping[str, str], *, because: str) -> float:
    raw = _required_str(name, env, because=because)
    try:
        return float(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be a number, got {raw!r}") from error


def _collect_required_int(
    name: str, env: Mapping[str, str], *, because: str, errors: list[str]
) -> int | None:
    try:
        return _required_int(name, env, because=because)
    except WorkerConfigError as error:
        errors.append(str(error))
        return None


def _collect_required_float(
    name: str, env: Mapping[str, str], *, because: str, errors: list[str]
) -> float | None:
    try:
        return _required_float(name, env, because=because)
    except WorkerConfigError as error:
        errors.append(str(error))
        return None


def _optional_int(name: str, env: Mapping[str, str]) -> int | None:
    raw = env.get(name, "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error


def _optional_float(name: str, env: Mapping[str, str]) -> float | None:
    raw = env.get(name, "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be a number, got {raw!r}") from error


def _warn_if_env_ignored(name: str, env: Mapping[str, str], *, reason: str) -> None:
    if env.get(name, "").strip():
        LOGGER.warning("%s is set but ignored: %s", name, reason)


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
    artifact_dir_raw = env.get(ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV, "").strip()
    is_default = not artifact_dir_raw

    if is_default:
        artifact_dir = _DEFAULT_ARTIFACT_DIR
        window_env = _optional_int(ML_WORKER_FALL_MODEL_WINDOW_ENV, env)
        stride_env = _optional_int(ML_WORKER_FALL_MODEL_STRIDE_ENV, env)
        operating_threshold_env = _optional_float(ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV, env)
        window = _DEFAULT_WINDOW if window_env is None else window_env
        stride = _DEFAULT_STRIDE if stride_env is None else stride_env
        operating_threshold = (
            _DEFAULT_OPERATING_THRESHOLD
            if operating_threshold_env is None
            else operating_threshold_env
        )
        operating_threshold_source = (
            "packaged manifest default" if operating_threshold_env is None else "env"
        )
        schema_version: int | None = _DEFAULT_SCHEMA_VERSION
        preprocessing_identity: str | None = _DEFAULT_PREPROCESSING_IDENTITY
        _warn_if_env_ignored(
            ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
            env,
            reason=(
                f"only read when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is also set; "
                "the packaged default model's own manifest value is used instead"
            ),
        )
        _warn_if_env_ignored(
            ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
            env,
            reason=(
                f"only read when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is also set; "
                "the packaged default model's own manifest value is used instead"
            ),
        )
    else:
        artifact_dir = artifact_dir_raw
        because = f"when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is set"
        errors: list[str] = []
        window = _collect_required_int(
            ML_WORKER_FALL_MODEL_WINDOW_ENV, env, because=because, errors=errors
        )
        stride = _collect_required_int(
            ML_WORKER_FALL_MODEL_STRIDE_ENV, env, because=because, errors=errors
        )
        operating_threshold = _collect_required_float(
            ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV, env, because=because, errors=errors
        )
        if errors:
            raise WorkerConfigError(
                f"{len(errors)} fall model environment variable(s) invalid: " + "; ".join(errors)
            )
        assert window is not None
        assert stride is not None
        assert operating_threshold is not None
        operating_threshold_source = "env"
        schema_version = _optional_int(ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV, env)
        preprocessing_identity = (
            env.get(ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV, "").strip() or None
        )

    model_type = env.get(ML_WORKER_FALL_MODEL_TYPE_ENV, "").strip() or _DEFAULT_TYPE
    weights = env.get(ML_WORKER_FALL_MODEL_WEIGHTS_ENV, "").strip() or _DEFAULT_WEIGHTS
    architecture = (
        env.get(ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV, "").strip() or _DEFAULT_ARCHITECTURE
    )
    LOGGER.info(
        "fall model operating_threshold resolved to %s (source: %s)",
        operating_threshold,
        operating_threshold_source,
    )
    framework: Literal["pytorch", "onnxruntime"] = (
        "onnxruntime" if env.get("ML_WORKER_PROFILE", "").strip() == "flow" else "pytorch"
    )
    try:
        return FallModelConfig(
            type=model_type,
            framework=framework,
            mode="sequence",
            artifact_dir=Path(artifact_dir),
            weights=weights,
            architecture=architecture,
            window=window,
            stride=stride,
            input_shape=(window, 56),
            operating_threshold=operating_threshold,
            schema_version=schema_version,
            preprocessing_identity=preprocessing_identity,
        )
    except ValidationError as error:
        if is_default:
            raise WorkerConfigError(
                "packaged default pose+bbox56 fall model is not fully provisioned at "
                f"{artifact_dir!r} ({error}); {_FETCH_MODELS_HINT}"
            ) from error
        raise WorkerConfigError(f"invalid fall model environment configuration: {error}") from error


def selected_fall_bundle_config_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    selection_path: Path | None = None,
    models_root: Path | None = None,
) -> SelectedFallBundleConfig | None:
    selection_path = FALL_SELECTION_PATH if selection_path is None else selection_path
    models_root = FALL_MODELS_ROOT if models_root is None else models_root
    if not selection_path.exists():
        return None
    try:
        raw_selection = selection_path.read_bytes()
        selection_document = json.loads(raw_selection)
        canonical_selection = json.dumps(
            selection_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if raw_selection != canonical_selection:
            raise WorkerConfigError("fall selection must be canonical JSON")
        desired = desired_model_bundle_from_selection_document(selection_document)
        return SelectedFallBundleConfig(
            models_root=models_root,
            desired=desired,
        )
    except (OSError, ValueError, TypeError, ModelBundleAdmissionError) as error:
        raise WorkerConfigError(f"invalid fall selection: {error}") from error


def worker_models_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> WorkerModelsConfig:
    selected = selected_fall_bundle_config_from_environment(environ)
    if selected is not None:
        return WorkerModelsConfig(selected=selected)
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
    if environment_models.selected is not None:
        if yaml_models is not None:
            raise WorkerConfigError(
                "selected fall bundle cannot coexist with a packaged fall model"
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
    "FALL_MODELS_ROOT",
    "FALL_SELECTION_PATH",
    "ML_WORKER_CLIP_RECORDING_ENABLED_ENV",
    "ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV",
    "ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV",
    "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV",
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV",
    "ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV",
    "ML_WORKER_FALL_MODEL_STRIDE_ENV",
    "ML_WORKER_FALL_MODEL_TYPE_ENV",
    "ML_WORKER_FALL_MODEL_WEIGHTS_ENV",
    "ML_WORKER_FALL_MODEL_WINDOW_ENV",
    "WORKER_REPLAY_TRACE_DIR_ENV",
    "clip_recording_config_from_environment",
    "fall_model_config_from_environment",
    "reject_retired_worker_environment",
    "replay_trace_directory_from_environment",
    "resolve_local_overrides",
    "selected_fall_bundle_config_from_environment",
    "worker_models_config_from_environment",
]
