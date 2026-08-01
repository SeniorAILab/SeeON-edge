from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worker.runtime.lease import GPU_LEASE_FILENAME, GpuLease, GpuLeaseUnavailableError


def test_second_process_is_refused_while_first_holds_gpu_lease(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from worker.runtime.lease import GpuLease

with GpuLease.acquire(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"

    try:
        with pytest.raises(GpuLeaseUnavailableError):
            _ = GpuLease.acquire(tmp_path)
    finally:
        process.terminate()
        _ = process.wait(timeout=5.0)


def test_clean_close_releases_the_gpu_lease_and_preserves_its_inode(tmp_path: Path) -> None:
    first = GpuLease.acquire(tmp_path)
    first.close()
    lease_path = tmp_path / GPU_LEASE_FILENAME
    before = lease_path.stat().st_ino

    with GpuLease.acquire(tmp_path) as second:
        during = lease_path.stat().st_ino
        assert second.held

    assert during == before
    assert lease_path.exists()
