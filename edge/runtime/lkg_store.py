from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from contracts.worker_config import PulledWorkerConfig

ML_WORKER_STATE_DIR_ENV = "ML_WORKER_STATE_DIR"
_DEFAULT_STATE_DIR = "/var/lib/ml-worker"
_LKG_FILENAME = "last-known-good.json"


def lkg_path() -> Path:
    return Path(os.environ.get(ML_WORKER_STATE_DIR_ENV, _DEFAULT_STATE_DIR)) / _LKG_FILENAME


def save_lkg(cfg: PulledWorkerConfig) -> None:
    path = lkg_path()
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(cfg.as_dict()), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"worker LKG save failed: {exc}", file=sys.stderr)


def load_lkg() -> PulledWorkerConfig | None:
    try:
        return PulledWorkerConfig.from_dict(json.loads(lkg_path().read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"worker LKG load skipped: {exc}", file=sys.stderr)
        return None


__all__ = ["ML_WORKER_STATE_DIR_ENV", "lkg_path", "save_lkg", "load_lkg"]
