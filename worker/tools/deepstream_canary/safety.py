"""Host-side live-stack protection and first-fault persistence."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from worker.tools.deepstream_canary.report import canonical_json

PROJECT_LABEL: Final = "com.docker.compose.project"
PROJECT_NAME: Final = "seeon-ds-canary"


@dataclass(frozen=True, slots=True)
class LiveContainer:
    container_id: str
    restart_count: int
    mounts: tuple[Path, ...]


class _Heartbeat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    status: str


class _RuntimeCamera(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    stale: bool


class _ClipRecorder(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    dropped_frames: int
    dropped_events: int


class _RuntimeStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    cameras: dict[str, _RuntimeCamera]
    clip_recorder: _ClipRecorder


class _LiveStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    cameras: dict[str, _Heartbeat]
    runtime: _RuntimeStatus


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    containers: tuple[LiveContainer, ...]
    xid_count: int
    gpu_used_mib: float = 0
    gpu_total_mib: float = 0
    gpu_utilization: float = 0
    gpu_processes: tuple[str, ...] = ()
    online_camera_ids: tuple[str, ...] = ()
    healthy_runtime_camera_ids: tuple[str, ...] = ()
    evidence_drops: int = 0


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    minimum_gpu_slack_mib: float
    maximum_gpu_utilization: float
    require_live_status: bool


@dataclass(frozen=True, slots=True)
class CanarySafetyError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def _run(command: tuple[str, ...]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def refuse_existing_project() -> None:
    ids = _run(
        (
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={PROJECT_LABEL}={PROJECT_NAME}",
            "--format",
            "{{.ID}}",
        )
    )
    if ids:
        raise CanarySafetyError("canary_project_exists", ids.replace("\n", ","))


def _container_snapshot(container_id: str) -> LiveContainer:
    restart_count = int(_run(("docker", "inspect", "-f", "{{.RestartCount}}", container_id)))
    mount_text = _run(
        (
            "docker",
            "inspect",
            "-f",
            "{{range .Mounts}}{{println .Source}}{{end}}",
            container_id,
        )
    )
    mounts = tuple(Path(line).resolve() for line in mount_text.splitlines() if line)
    return LiveContainer(container_id=container_id, restart_count=restart_count, mounts=mounts)


def _gpu_snapshot() -> tuple[float, float, float, tuple[str, ...]]:
    gpu = _run(
        (
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    fields = tuple(item.strip() for item in gpu.splitlines()[0].split(","))
    if len(fields) != 3:
        raise CanarySafetyError("gpu_telemetry_missing", gpu)
    processes = _run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    return float(fields[0]), float(fields[1]), float(fields[2]), tuple(processes.splitlines())


def _live_status() -> tuple[tuple[str, ...], tuple[str, ...], int]:
    completed = subprocess.run(
        ("curl", "--fail", "--silent", "--show-error", "http://127.0.0.1:8000/api/v1/status"),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return (), (), -1
    status = _LiveStatus.model_validate_json(completed.stdout)
    online = tuple(sorted(key for key, item in status.cameras.items() if item.status == "online"))
    healthy = tuple(sorted(key for key, item in status.runtime.cameras.items() if not item.stale))
    recorder = status.runtime.clip_recorder
    return online, healthy, recorder.dropped_frames + recorder.dropped_events


def capture_live_snapshot() -> LiveSnapshot:
    ids = _run(
        (
            "docker",
            "ps",
            "--filter",
            f"label={PROJECT_LABEL}!={PROJECT_NAME}",
            "--format",
            "{{.ID}}",
        )
    )
    containers = tuple(_container_snapshot(item) for item in ids.splitlines() if item)
    kernel = subprocess.run(
        ("journalctl", "--boot", "0", "--dmesg", "--no-pager"),
        check=False,
        capture_output=True,
        text=True,
    )
    xid_count = kernel.stdout.count("NVRM: Xid") if kernel.returncode == 0 else -1
    gpu_used, gpu_total, gpu_utilization, gpu_processes = _gpu_snapshot()
    online, healthy, evidence_drops = _live_status()
    return LiveSnapshot(
        containers=containers,
        xid_count=xid_count,
        gpu_used_mib=gpu_used,
        gpu_total_mib=gpu_total,
        gpu_utilization=gpu_utilization,
        gpu_processes=gpu_processes,
        online_camera_ids=online,
        healthy_runtime_camera_ids=healthy,
        evidence_drops=evidence_drops,
    )


def refuse_mount_overlap(canary_mounts: tuple[Path, ...], snapshot: LiveSnapshot) -> None:
    for canary in canary_mounts:
        resolved = canary.resolve()
        for container in snapshot.containers:
            for live in container.mounts:
                if resolved == live or resolved in live.parents or live in resolved.parents:
                    raise CanarySafetyError(
                        "live_mount_intersection", f"{resolved} intersects {live}"
                    )


def compare_live_snapshot(before: LiveSnapshot, limits: SafetyLimits) -> None:
    after = capture_live_snapshot()
    baseline = {item.container_id: item for item in before.containers}
    current = {item.container_id: item for item in after.containers}
    if current != baseline:
        raise CanarySafetyError(
            "live_container_changed", "container identity/restart/mount changed"
        )
    if before.xid_count < 0 or after.xid_count < 0:
        raise CanarySafetyError("kernel_fault_monitor_unavailable", "current-boot log unavailable")
    if after.xid_count > before.xid_count:
        raise CanarySafetyError("new_xid", f"before={before.xid_count},after={after.xid_count}")
    if after.gpu_total_mib - after.gpu_used_mib < limits.minimum_gpu_slack_mib:
        raise CanarySafetyError("gpu_slack_breach", str(after.gpu_total_mib - after.gpu_used_mib))
    if after.gpu_utilization > limits.maximum_gpu_utilization:
        raise CanarySafetyError("gpu_utilization_breach", str(after.gpu_utilization))
    baseline_pids = {item.split(",", maxsplit=1)[0] for item in before.gpu_processes}
    current_pids = {item.split(",", maxsplit=1)[0] for item in after.gpu_processes}
    if not baseline_pids <= current_pids:
        raise CanarySafetyError(
            "live_gpu_process_missing", str(sorted(baseline_pids - current_pids))
        )
    if not set(before.online_camera_ids) <= set(after.online_camera_ids):
        raise CanarySafetyError("live_camera_stale", "online camera left online state")
    if not set(before.healthy_runtime_camera_ids) <= set(after.healthy_runtime_camera_ids):
        raise CanarySafetyError("live_camera_stale", "runtime camera became stale")
    if limits.require_live_status and (before.evidence_drops < 0 or after.evidence_drops < 0):
        raise CanarySafetyError("live_status_monitor_unavailable", "status endpoint unavailable")
    if before.evidence_drops >= 0 and after.evidence_drops > before.evidence_drops:
        raise CanarySafetyError("live_evidence_drop_increase", str(after.evidence_drops))


def persist_first_fault(path: Path, error: CanarySafetyError) -> None:
    value = {
        "schema_version": 1,
        "code": error.code,
        "detail": error.detail,
        "action": "canary_down_only",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as target:
        target.write(canonical_json(value))
        target.flush()
        os.fsync(target.fileno())
