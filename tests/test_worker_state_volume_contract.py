from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = (_ROOT / "compose.edge.yaml").read_text(encoding="utf-8")


def _service_block(name: str) -> str:
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)", _COMPOSE, re.MULTILINE | re.DOTALL
    )
    assert match, f"compose.edge.yaml declares no service named {name!r}"
    return match.group(1)


def _mount_target(block: str, volume: str) -> str:
    match = re.search(rf"-\s+{re.escape(volume)}:([^\s:]+)", block)
    assert match, f"service does not mount {volume!r}"
    return match.group(1)


def test_the_worker_is_told_to_use_its_mounted_state_volume() -> None:
    block = _service_block("ml-worker")
    mounted = _mount_target(block, "worker-local-state")

    match = re.search(r"-\s+--state-dir\n\s+-\s+(\S+)", block)
    assert match, (
        "ml-worker does not pass --state-dir, so it falls back to the home "
        "default and writes its durable delivery queue into the container's "
        "writable layer, where container replacement destroys pending evidence"
    )
    assert match.group(1) == mounted, (
        f"ml-worker is told to use {match.group(1)!r} but its worker-local-state "
        f"volume is mounted at {mounted!r}; the queue would not be on the volume"
    )


def test_the_worker_state_default_is_unsuitable_for_a_container() -> None:
    from worker.runtime.state_dir import resolve_state_dir

    resolved = resolve_state_dir()
    assert resolved.is_relative_to(Path.home()), (
        "resolve_state_dir no longer returns a home-relative path; re-examine "
        "whether the container still needs an explicit --state-dir"
    )


def test_the_refused_evidence_command_can_actually_be_run() -> None:
    block = _service_block("edge-refused-evidence")

    mounted = _mount_target(block, "worker-local-state")
    assert ":ro" not in block.split(mounted)[1].split("\n")[0], (
        "the operator service mounts the queue read-only, so requeue cannot write"
    )
    assert "scripts/ops/review-refused-evidence.py" in block, (
        "the service does not invoke the documented command"
    )
    assert f"- {mounted}" in block, "the command is not pointed at the volume the service mounts"
    assert 'profiles: ["ops"]' in block or "- ops" in block, (
        "a one-shot operator tool must not start with the stack"
    )
