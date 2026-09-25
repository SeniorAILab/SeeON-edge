"""Subprocess coverage for the fall-deploy.sh registry-resolve step.

`bash -n`/shellcheck cannot see the embedded python one-liner's runtime
behavior (an f-string with escaped quotes is valid-looking shell text but
invalid Python syntax), so this actually executes the script against a
stubbed `curl` to exercise both die paths before any download/docker side
effect: nothing before check_repo() touches docker, hf, or $E (confirmed by
reading fall-deploy.sh -- only curl + python3 run before it).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "research" / "fall-deploy.sh"


def _write_stub(path: Path, name: str, body: str) -> None:
    command = path / name
    command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)


def _run_fall_deploy(tmp_path: Path, *, model_version: dict) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fixture = tmp_path / "model_version.json"
    fixture.write_text(json.dumps({"model_version": model_version}), encoding="utf-8")
    _write_stub(bin_dir, "curl", 'cat "$MLFLOW_FIXTURE"')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MLFLOW_FIXTURE": str(fixture)}
    return subprocess.run(
        ["bash", str(SCRIPT), "2"], env=env, text=True, capture_output=True, check=False
    )


def test_fall_deploy_dies_on_missing_registry_tag(tmp_path: Path) -> None:
    result = _run_fall_deploy(
        tmp_path,
        model_version={
            "version": "2",
            "tags": [
                {"key": "hf_repo", "value": "Berom0227/seeon-fall-v1"},
                {"key": "hf_revision", "value": "abc123"},
            ],
        },
    )

    assert result.returncode != 0
    assert "registry version 2 lacks onnx_sha256" in result.stdout + result.stderr


def test_fall_deploy_dies_on_disallowed_hf_repo(tmp_path: Path) -> None:
    result = _run_fall_deploy(
        tmp_path,
        model_version={
            "version": "2",
            "tags": [
                {"key": "hf_repo", "value": "evil/x"},
                {"key": "hf_revision", "value": "abc123"},
                {"key": "onnx_sha256", "value": "deadbeef"},
            ],
        },
    )

    assert result.returncode != 0
    assert "hf_repo 'evil/x' does not match" in result.stdout + result.stderr
