"""Synchronous durable first-fault receipt for the isolated dark runner."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from worker.runtime.deepstream.config import ChildConfig


@dataclass(frozen=True, slots=True)
class DarkFirstFault:
    schema_version: int
    profile: str
    stage: str
    category: str
    exit_code: int
    worker_boot_id: str
    child_instance_id: str
    fault_time_iso: str
    action: str = "exit_container_retire_context"


def persist_first_fault(
    path: Path,
    *,
    category: str,
    exit_code: int,
    worker_boot_id: uuid.UUID,
    child_instance_id: uuid.UUID,
) -> DarkFirstFault:
    record = DarkFirstFault(
        schema_version=1,
        profile="nvidia",
        stage="deepstream_child",
        category=category,
        exit_code=exit_code,
        worker_boot_id=str(worker_boot_id),
        child_instance_id=str(child_instance_id),
        fault_time_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        return record
    encoded = json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        _ = temporary.write(encoded)
        temporary.flush()
        _ = os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    try:
        os.link(temporary_path, path)
    except FileExistsError:
        pass
    finally:
        temporary_path.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _ = os.fsync(directory)
    finally:
        os.close(directory)
    return record


def persist_child_fault(config: ChildConfig, category: str) -> DarkFirstFault:
    return persist_first_fault(
        config.first_fault_path,
        category=category,
        exit_code=4,
        worker_boot_id=config.worker_boot_id,
        child_instance_id=config.child_instance_id,
    )


__all__ = ["DarkFirstFault", "persist_child_fault", "persist_first_fault"]
