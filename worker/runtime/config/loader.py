from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import yaml
from pydantic import JsonValue, TypeAdapter, ValidationError

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import WorkerConfig

EDGE_CAMERA_CONFIG_ENV: Final = "EDGE_CAMERA_CONFIG"
ML_WORKER_DEV_MJPEG_ENV: Final = "ML_WORKER_DEV_MJPEG"
ML_WORKER_DEV_MJPEG_HOST_ENV: Final = "ML_WORKER_DEV_MJPEG_HOST"
ML_WORKER_DEV_MJPEG_PORT_ENV: Final = "ML_WORKER_DEV_MJPEG_PORT"
_YAML_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def load_worker_config(path: str | Path) -> WorkerConfig:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        raise WorkerConfigError(
            f"worker camera config must be YAML, JSON is not supported: {config_path}"
        )
    try:
        raw = _YAML_VALUE_ADAPTER.validate_python(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
        if not isinstance(raw, dict):
            raise WorkerConfigError(
                f"worker camera config must contain a YAML mapping: {config_path}"
            )
        return WorkerConfig.model_validate(raw)
    except OSError as error:
        raise WorkerConfigError(f"worker camera config not readable: {config_path}") from error
    except yaml.YAMLError as error:
        raise WorkerConfigError(f"worker camera config is not valid YAML: {config_path}") from error
    except ValidationError as error:
        fields = ", ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_url=False, include_input=False)
        )
        raise WorkerConfigError(f"worker camera config invalid: {fields}") from error


def resolve_config_path(value: str | None = None) -> Path:
    raw_path = value if value is not None else os.environ.get(EDGE_CAMERA_CONFIG_ENV, "")
    stripped = raw_path.strip()
    if not stripped:
        raise WorkerConfigError(
            f"worker camera config path required via --config or {EDGE_CAMERA_CONFIG_ENV}"
        )
    return Path(stripped)


__all__ = [
    "EDGE_CAMERA_CONFIG_ENV",
    "ML_WORKER_DEV_MJPEG_ENV",
    "ML_WORKER_DEV_MJPEG_HOST_ENV",
    "ML_WORKER_DEV_MJPEG_PORT_ENV",
    "load_worker_config",
    "resolve_config_path",
]
