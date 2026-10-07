from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.real_stack

_CUDA_SUCCESS = 0
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> ctypes.CDLL:
    library = ctypes.CDLL("libcuda.so.1")
    if library.cuInit(0) != _CUDA_SUCCESS:
        pytest.skip("no CUDA driver on this host")
    return library


def _primary_context_active(driver: ctypes.CDLL) -> bool:
    device = ctypes.c_int()
    if driver.cuDeviceGet(ctypes.byref(device), 0) != _CUDA_SUCCESS:
        pytest.skip("no CUDA device 0")
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    result = driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
    assert result == _CUDA_SUCCESS
    return bool(active.value)


def _current_context(driver: ctypes.CDLL) -> int | None:
    context = ctypes.c_void_p()
    assert driver.cuCtxGetCurrent(ctypes.byref(context)) == _CUDA_SUCCESS
    return context.value


def test_the_host_copy_path_makes_no_cuda_context_current() -> None:
    from worker.adapters.deepstream.tensor_rows import host_array_from_tensor

    driver = _driver()
    assert _current_context(driver) is None, "Python must hold no CUDA context before the copy"

    copied = host_array_from_tensor(np.zeros((2, 57), dtype=np.float32))
    assert copied.shape == (2, 57)

    assert _current_context(driver) is None, (
        "the tensor copy must not make a CUDA context current on the calling thread"
    )


_WORKER_IMPORTS_PROBE = """
import sys

import worker.adapters.deepstream.service_maker
import worker.adapters.model.ort_bed_seg
import worker.adapters.model.ort_pose_bbox56
import worker.runtime.flow

print(",".join(name for name in ("torch", "cupy") if name in sys.modules))
"""


def test_the_worker_process_imports_no_other_cuda_client() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", _WORKER_IMPORTS_PROBE],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "", f"worker imports load another CUDA client: {probe.stdout}"


def test_ort_runs_on_cpu_and_leaves_the_gpu_alone() -> None:
    import onnxruntime

    driver = _driver()
    before = _primary_context_active(driver)
    session_providers = onnxruntime.get_available_providers()
    assert "CPUExecutionProvider" in session_providers
    assert "CUDAExecutionProvider" not in session_providers, (
        "the flow image must not ship a CUDA execution provider"
    )
    assert _primary_context_active(driver) is before
