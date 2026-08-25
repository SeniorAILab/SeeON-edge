"""Real Docker execution with a host watchdog and canary-only teardown."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from worker.tools.deepstream_canary.safety import (
    CanarySafetyError,
    LiveSnapshot,
    SafetyLimits,
    compare_live_snapshot,
    persist_first_fault,
)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    compose_path: Path
    evidence_dir: Path
    baseline: LiveSnapshot
    monitor_seconds: int
    relay_token: str
    safety_limits: SafetyLimits


def _compose(request: ExecutionRequest, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            "docker",
            "compose",
            "-p",
            "seeon-ds-canary",
            "-f",
            str(request.compose_path),
            *arguments,
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CANARY_RELAY_TOKEN": request.relay_token},
    )


def _generate_corpus(root: Path) -> Path:
    corpus = root / "run" / "scratch" / "loopback.mp4"
    completed = subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=15",
            "-t",
            "60",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            str(corpus),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CanarySafetyError("corpus_generation_failed", completed.stderr[-500:])
    _ = corpus.chmod(0o444)
    _ = corpus.parent.chmod(0o755)
    return corpus


def execute_canary(request: ExecutionRequest) -> int:
    """Execute only the isolated project; any first fault tears down only it."""
    fault_path = request.evidence_dir / "first-fault.json"
    stop = threading.Event()
    stack_started = threading.Event()
    watchdog_fault: list[CanarySafetyError] = []

    def abort(error: CanarySafetyError) -> None:
        watchdog_fault.append(error)
        stop.set()
        _ = _compose(request, "down", "--remove-orphans")

    def watchdog() -> None:
        while not stop.wait(10.0):
            if stack_started.is_set():
                exited = _compose(request, "ps", "--status", "exited", "--services")
                failed_services = tuple(
                    service
                    for service in exited.stdout.splitlines()
                    if service and service != "engine-builder"
                )
                if exited.returncode != 0 or failed_services:
                    abort(CanarySafetyError("canary_service_exited", ",".join(failed_services)))
                    return
            try:
                compare_live_snapshot(request.baseline, request.safety_limits)
            except CanarySafetyError as error:
                abort(error)
                return

    try:
        _ = _generate_corpus(request.evidence_dir)
        monitor = threading.Thread(target=watchdog, name="canary-live-watchdog", daemon=True)
        monitor.start()
        up = _compose(request, "up", "-d", "--remove-orphans")
        stack_started.set()
        (request.evidence_dir / "compose-up.log").write_text(
            up.stdout + up.stderr, encoding="utf-8"
        )
        if up.returncode != 0:
            logs = _compose(request, "logs", "--no-color")
            _ = (request.evidence_dir / "compose-failure.log").write_text(
                logs.stdout + logs.stderr, encoding="utf-8"
            )
            error = CanarySafetyError("compose_up_failed", up.stderr[-500:])
            persist_first_fault(fault_path, error)
            return 1
        _ = stop.wait(request.monitor_seconds)
        stop.set()
        monitor.join(timeout=15.0)
        if watchdog_fault:
            raise watchdog_fault[0]
        compare_live_snapshot(request.baseline, request.safety_limits)
        logs = _compose(request, "logs", "--no-color")
        (request.evidence_dir / "compose.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        execution = {
            "schema_version": 1,
            "status": "collected",
            "note": "verdict requires independent raw-rung receipts",
        }
        (request.evidence_dir / "execution.json").write_text(
            json.dumps(execution, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except CanarySafetyError as error:
        logs = _compose(request, "logs", "--no-color")
        _ = (request.evidence_dir / "compose-failure.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        persist_first_fault(fault_path, error)
        return 1
    else:
        return 0
    finally:
        stop.set()
        down = _compose(request, "down", "--remove-orphans")
        (request.evidence_dir / "compose-down.log").write_text(
            down.stdout + down.stderr, encoding="utf-8"
        )
