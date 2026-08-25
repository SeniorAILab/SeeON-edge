"""Real Docker execution with a host watchdog and canary-only teardown."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from worker.tools.deepstream_canary.collector import CollectionRequest, collect_recorded_telemetry
from worker.tools.deepstream_canary.execution_artifacts import (
    ExecutionArtifactSources,
    ReceiptEmissionRequest,
    emit_receipts,
    generate_corpus,
    gpu_sample,
    native_windows,
)
from worker.tools.deepstream_canary.models import CanaryMode
from worker.tools.deepstream_canary.safety import (
    CanarySafetyError,
    LiveSnapshot,
    SafetyLimits,
    compare_live_snapshot,
    persist_first_fault,
)
from worker.tools.deepstream_canary.telemetry import RuntimeGpuSample


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    compose_path: Path
    evidence_dir: Path
    baseline: LiveSnapshot
    rung_durations: tuple[tuple[str, int], ...]
    publisher_count: int
    relay_token: str
    safety_limits: SafetyLimits
    mode: CanaryMode
    rungs: tuple[str, ...]
    artifacts: ExecutionArtifactSources


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


def _require_success(
    completed: subprocess.CompletedProcess[str],
    code: str,
    detail: str,
) -> None:
    if completed.returncode != 0:
        raise CanarySafetyError(code, detail)


def _require_sample(sample: RuntimeGpuSample | None, rung: str) -> RuntimeGpuSample:
    if sample is None:
        raise CanarySafetyError("native_child_not_ready", rung)
    return sample


def execute_canary(request: ExecutionRequest) -> int:
    """Execute only the isolated project; any first fault tears down only it."""
    fault_path = request.evidence_dir / "first-fault.json"
    stop = threading.Event()
    stack_started = threading.Event()
    capacity_armed = threading.Event()
    active_rung: list[str] = ["prepare"]
    gpu_samples: dict[str, list[RuntimeGpuSample]] = {}
    sample_lock = threading.Lock()
    watchdog_fault: list[CanarySafetyError] = []

    def abort(error: CanarySafetyError) -> None:
        watchdog_fault.append(error)
        stop.set()
        persist_first_fault(fault_path, error)
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
                snapshot = compare_live_snapshot(
                    request.baseline,
                    replace(
                        request.safety_limits,
                        enforce_gpu_capacity=capacity_armed.is_set(),
                    ),
                )
                if capacity_armed.is_set() and stack_started.is_set():
                    sample = gpu_sample(snapshot)
                    if sample is not None:
                        with sample_lock:
                            gpu_samples.setdefault(active_rung[0], []).append(sample)
            except CanarySafetyError as error:
                abort(error)
                return

    try:
        corpus = generate_corpus(request.evidence_dir)
        monitor = threading.Thread(target=watchdog, name="canary-live-watchdog", daemon=True)
        monitor.start()
        prepare = _compose(request, "run", "--rm", "engine-builder")
        _ = (request.evidence_dir / "engine-prepare.log").write_text(
            prepare.stdout + prepare.stderr, encoding="utf-8"
        )
        if watchdog_fault:
            raise watchdog_fault[0]
        _require_success(prepare, "engine_prepare_failed", prepare.stderr[-500:])
        capacity_armed.set()
        phase_logs: list[str] = []
        for rung, duration in request.rung_durations:
            stack_started.clear()
            active_rung[0] = rung
            selected = "zero" if rung == "zero" else "workload"
            (request.evidence_dir / "raw" / "active-config").write_text(
                f"{selected}\n", encoding="utf-8"
            )
            if selected == "workload":
                publishers = tuple(
                    f"publisher-{index:02d}"
                    for index in range(1, request.publisher_count + 1)
                )
                publisher_up = _compose(request, "up", "-d", *publishers)
                phase_logs.append(publisher_up.stdout + publisher_up.stderr)
                _require_success(publisher_up, "publisher_start_failed", rung)
                up = _compose(request, "up", "-d", "--force-recreate", "ml-worker")
            else:
                up = _compose(request, "up", "-d", "relay-stub", "mediamtx", "ml-worker")
            phase_logs.append(up.stdout + up.stderr)
            stack_started.set()
            if watchdog_fault:
                raise watchdog_fault[0]
            _require_success(up, "compose_up_failed", up.stderr[-500:])
            _ = stop.wait(duration)
            if watchdog_fault:
                raise watchdog_fault[0]
            final_snapshot = compare_live_snapshot(
                request.baseline, request.safety_limits
            )
            final_sample = _require_sample(gpu_sample(final_snapshot), rung)
            with sample_lock:
                gpu_samples.setdefault(rung, []).append(final_sample)
                phase_gpu = tuple(gpu_samples[rung])
            camera_count = 0 if rung == "zero" else request.publisher_count
            _ = collect_recorded_telemetry(
                CollectionRequest(
                    evidence_dir=request.evidence_dir,
                    rung=rung,
                    mode=request.mode,
                    clean_steady_seconds=duration,
                    camera_count=camera_count,
                    gpu_samples=phase_gpu,
                    native_windows=(
                        ()
                        if rung == "zero"
                        else native_windows(
                            request.evidence_dir / "raw" / "native-telemetry.jsonl"
                        )
                    ),
                )
            )
            stack_started.clear()
            stopped = _compose(request, "stop", "ml-worker")
            phase_logs.append(stopped.stdout + stopped.stderr)
        stop.set()
        monitor.join(timeout=15.0)
        _ = (request.evidence_dir / "compose-up.log").write_text(
            "".join(phase_logs), encoding="utf-8"
        )
        _ = emit_receipts(
            ReceiptEmissionRequest(
                evidence_dir=request.evidence_dir,
                rungs=request.rungs,
                artifacts=request.artifacts,
            ),
            corpus,
        )
        logs = _compose(request, "logs", "--no-color")
        (request.evidence_dir / "compose.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        execution = {
            "schema_version": 1,
            "status": "collected",
            "note": "raw rung receipts emitted for independent verification",
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
        if not watchdog_fault:
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
