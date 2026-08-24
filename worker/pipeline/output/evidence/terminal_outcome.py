"""Publish-once terminal clip outcomes committed strictly after the manifest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import final

from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file


class TerminalClipState(StrEnum):
    READY = "ClipCommitted"
    UNAVAILABLE = "ClipUnavailableCommitted"
    CORRUPT = "ClipCorruptCommitted"


@dataclass(frozen=True, slots=True)
class TerminalClipOutcome:
    clip_id: str
    event_ids: tuple[str, ...]
    state: TerminalClipState
    manifest_sha256: str


@final
class TerminalOutcomeConflictError(Exception):
    pass


def commit_terminal_outcome(clip_dir: Path, outcome: TerminalClipOutcome) -> Path:
    payload = json.dumps(
        {
            "clip_id": outcome.clip_id,
            "event_ids": list(outcome.event_ids),
            "manifest_sha256": outcome.manifest_sha256,
            "state": outcome.state.value,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode() + b"\n"
    destination = clip_dir / "terminal-outcome.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise TerminalOutcomeConflictError
        fsync_file(destination)
        return destination
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("xb") as output:
        _ = output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)
    fsync_file(destination)
    fsync_directory(clip_dir)
    return destination


__all__ = [
    "TerminalClipOutcome",
    "TerminalClipState",
    "TerminalOutcomeConflictError",
    "commit_terminal_outcome",
]
