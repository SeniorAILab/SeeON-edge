"""C5 native child supervision contracts."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

import pytest

import worker.runtime.deepstream.supervisor as supervisor
from worker.runtime.lease import GpuLease, GpuLeaseUnavailableError


def test_dark_mode_constructs_no_child_by_default() -> None:
    # Given / When
    # Then
    assert supervisor.configured_dark_supervisors({}) == ()


def test_c5_admits_only_one_explicit_gpu_zero_child() -> None:
    # Given
    environment = {
        "SEEON_DEEPSTREAM_DARK_CHILD": "1",
        "NVIDIA_VISIBLE_DEVICES": "0",
    }

    # When
    configured = supervisor.configured_dark_supervisors(environment)

    # Then
    assert tuple(item.gpu_id for item in configured) == ("0",)
    with pytest.raises(supervisor.DarkChildConfigError) as failed:
        _ = supervisor.configured_dark_supervisors(
            {"SEEON_DEEPSTREAM_DARK_CHILD": "1", "NVIDIA_VISIBLE_DEVICES": "0,2"}
        )
    assert failed.value.code == "unsupported_gpu"


def test_supervisor_uses_ready_fd_not_misleading_stdout(tmp_path: Path) -> None:
    # Given
    executable = tmp_path / "misleading-child"
    _ = executable.write_text(
        "#!/bin/sh\nprintf 'READY\\n'\nexec sleep 2\n",
        encoding="utf-8",
    )
    _ = executable.chmod(0o700)
    child = supervisor.DeepStreamChildSupervisor(
        supervisor.ChildConfig(
            executable=executable,
            gpu_id="0",
            worker_boot_id=uuid.uuid4(),
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "first-fault.bin",
            startup_timeout_sec=0.1,
            stop_timeout_sec=0.2,
        )
    )

    # When / Then
    with pytest.raises(supervisor.ChildStartupError) as failed:
        child.start()
    assert failed.value.code == "ready_failed"
    child.stop()
    child.stop()
    assert child.pid is None


def test_supervisor_refuses_before_spawn_when_python_gpu_lease_is_held(tmp_path: Path) -> None:
    # Given
    lease_state = tmp_path / "lease"
    child = supervisor.DeepStreamChildSupervisor(
        supervisor.ChildConfig(
            executable=tmp_path / "must-not-spawn",
            gpu_id="0",
            worker_boot_id=uuid.uuid4(),
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "fault",
            lease_state_dir=lease_state,
        )
    )

    # When / Then
    with GpuLease.acquire(lease_state):
        with pytest.raises(GpuLeaseUnavailableError):
            child.start()
    assert child.pid is None


def test_unsafe_stale_socket_is_rejected_without_unlinking(tmp_path: Path) -> None:
    # Given
    socket_dir = tmp_path / "ipc"
    socket_dir.mkdir(mode=0o700)
    stale = socket_dir / "control.sock"
    _ = stale.write_bytes(b"stale")
    os.chmod(stale, 0o600)

    # When / Then
    with pytest.raises(supervisor.PrivatePathError) as failed:
        supervisor.remove_stale_socket(stale)
    assert failed.value.code == "stale_socket_unsafe"
    assert stale.exists()
    assert stat.S_IMODE(socket_dir.stat().st_mode) == 0o700
