from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from contracts.worker_config import PulledWorkerConfig
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
from worker.runtime.config.restart import RestartDirective

ML_WORKER_STATE_DIR_ENV: Final = "ML_WORKER_STATE_DIR"
DEFAULT_STATE_DIR: Final = "/var/lib/ml-worker"
WORKER_CONFIG_LKG_FILENAME: Final = "worker-config-last-known-good.json"
PULLED_CONFIG_LKG_FILENAME: Final = "last-known-good.json"

JsonObject: TypeAlias = dict[str, JsonValue]


class _LkgEnvelope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    generation: int
    version: int
    registry_version: int
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class StoredConfigPayload:
    payload: JsonObject
    directive: RestartDirective
    registry_version: int


class WorkerConfigLkgStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = worker_config_lkg_path() if path is None else path
        self._lock: threading.Lock = threading.Lock()

    def save(self, payload: JsonObject, directive: RestartDirective) -> bool:
        registry_version = _registry_version(payload)
        with self._lock:
            current = self._load_unlocked(report_error=False)
            if current is not None and (
                directive,
                registry_version,
            ) < (
                current.directive,
                current.registry_version,
            ):
                return False
            envelope = _LkgEnvelope(
                generation=directive.generation,
                version=directive.version,
                registry_version=registry_version,
                payload=payload,
            )
            try:
                _atomic_write(self.path, envelope.model_dump_json())
            except OSError:
                print("worker config LKG save failed", file=sys.stderr)
                return False
            return True

    def load(self) -> StoredConfigPayload | None:
        with self._lock:
            return self._load_unlocked(report_error=True)

    def _load_unlocked(self, *, report_error: bool) -> StoredConfigPayload | None:
        try:
            envelope = _LkgEnvelope.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            if report_error:
                print("worker config LKG load skipped", file=sys.stderr)
            return None
        return StoredConfigPayload(
            payload=envelope.payload,
            directive=RestartDirective(
                generation=envelope.generation,
                version=envelope.version,
            ),
            registry_version=envelope.registry_version,
        )


_pulled_lkg_lock = threading.Lock()


def worker_config_lkg_path() -> Path:
    return _state_dir() / WORKER_CONFIG_LKG_FILENAME


def lkg_path() -> Path:
    return _state_dir() / PULLED_CONFIG_LKG_FILENAME


def save_lkg(config: PulledWorkerConfig) -> None:
    with _pulled_lkg_lock:
        try:
            _atomic_write(
                lkg_path(),
                json.dumps(config.as_dict(), separators=(",", ":"), sort_keys=True),
            )
        except OSError:
            print("worker LKG save failed", file=sys.stderr)


def load_lkg() -> PulledWorkerConfig | None:
    with _pulled_lkg_lock:
        try:
            return BackendWorkerConfigPayload.model_validate_json(
                lkg_path().read_text(encoding="utf-8")
            ).to_pulled_config()
        except (OSError, ValidationError):
            print("worker LKG load skipped", file=sys.stderr)
            return None


def _state_dir() -> Path:
    return Path(os.environ.get(ML_WORKER_STATE_DIR_ENV, DEFAULT_STATE_DIR))


def _registry_version(payload: JsonObject) -> int:
    value = payload.get("registry_version", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_STATE_DIR",
    "ML_WORKER_STATE_DIR_ENV",
    "PULLED_CONFIG_LKG_FILENAME",
    "WORKER_CONFIG_LKG_FILENAME",
    "JsonObject",
    "JsonValue",
    "StoredConfigPayload",
    "WorkerConfigLkgStore",
    "lkg_path",
    "load_lkg",
    "save_lkg",
    "worker_config_lkg_path",
]
