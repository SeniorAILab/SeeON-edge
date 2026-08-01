from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from worker.adapters.device.cuda.probe import CudaCapability, probe_cuda_capability


class _FakeCuda:
    def __init__(
        self,
        *,
        arch_list: Callable[[], tuple[str, ...]] | Exception = (),
        device_count: Callable[[], int] | Exception = 0,
        is_available: Callable[[], bool] | Exception = False,
    ) -> None:
        self._arch_list = arch_list
        self._device_count = device_count
        self._is_available = is_available

    def get_arch_list(self) -> tuple[str, ...]:
        if isinstance(self._arch_list, Exception):
            raise self._arch_list
        return self._arch_list

    def device_count(self) -> int:
        if isinstance(self._device_count, Exception):
            raise self._device_count
        return self._device_count

    def is_available(self) -> bool:
        if isinstance(self._is_available, Exception):
            raise self._is_available
        return self._is_available


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


def test_cuda_capability_true_when_available() -> None:
    # Given -- a healthy CUDA install: driver initializes, kernels compiled in
    fake_torch = _FakeTorch(
        _FakeCuda(arch_list=("sm_90",), device_count=1, is_available=True)
    )

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability == CudaCapability(
        available=True, reason="cuda available", device_count=1, arch_list=("sm_90",)
    )


def test_cuda_capability_false_when_torch_import_fails() -> None:
    def failing_importer() -> Any:
        raise ImportError("no module named torch")

    # When
    capability = probe_cuda_capability(importer=failing_importer)

    # Then -- fail closed, never raises past the probe boundary
    assert capability.available is False
    assert "torch import failed" in capability.reason
    assert capability.device_count == 0
    assert capability.arch_list == ()


def test_cuda_capability_false_when_no_devices_visible() -> None:
    # Given -- the real signal on this repo's macOS dev machines: no NVML devices
    fake_torch = _FakeTorch(_FakeCuda(arch_list=(), device_count=0, is_available=False))

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability.available is False
    assert capability.reason == "torch.cuda.is_available() is False and no CUDA devices are visible"


def test_cuda_capability_false_diagnoses_broken_wheel_missing_arch_kernels() -> None:
    # Given -- NVML sees a device (device_count > 0) but the torch wheel has no
    # compiled kernels for it (empty arch_list): the ADR-0002 root cause.
    fake_torch = _FakeTorch(_FakeCuda(arch_list=(), device_count=1, is_available=False))

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability.available is False
    assert capability.device_count == 1
    assert "no compiled arch kernels" in capability.reason
    assert "Blackwell sm_120" in capability.reason


def test_cuda_capability_false_reports_devices_visible_but_unavailable_with_arch() -> None:
    # Given -- devices visible, kernels compiled in, but is_available() still False
    # (e.g. driver/runtime mismatch) -- the remaining diagnostic branch.
    fake_torch = _FakeTorch(_FakeCuda(arch_list=("sm_90",), device_count=1, is_available=False))

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability.available is False
    assert "despite 1 device(s) visible" in capability.reason
    assert "sm_90" in capability.reason


def test_cuda_capability_false_when_is_available_raises() -> None:
    fake_torch = _FakeTorch(
        _FakeCuda(arch_list=("sm_90",), device_count=1, is_available=RuntimeError("driver error"))
    )

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then -- backend probe failure is reported, not raised
    assert capability.available is False
    assert "torch.cuda.is_available() raised" in capability.reason
    # arch_list/device_count gathered before the failing call are preserved
    assert capability.device_count == 1
    assert capability.arch_list == ("sm_90",)


def test_cuda_capability_defaults_arch_list_when_get_arch_list_raises() -> None:
    # Given -- get_arch_list() itself raises; must not break the whole probe
    fake_torch = _FakeTorch(
        _FakeCuda(arch_list=RuntimeError("no arch"), device_count=0, is_available=False)
    )

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability.arch_list == ()
    assert capability.available is False


def test_cuda_capability_defaults_device_count_when_device_count_raises() -> None:
    # Given -- device_count() itself raises; must not break the whole probe
    fake_torch = _FakeTorch(
        _FakeCuda(arch_list=(), device_count=RuntimeError("no count"), is_available=False)
    )

    # When
    capability = probe_cuda_capability(importer=lambda: fake_torch)

    # Then
    assert capability.device_count == 0
    assert capability.available is False


def test_probe_cuda_capability_against_real_torch_is_honest_on_this_dev_machine() -> None:
    """No fakes: exercises the real ``import torch`` default on this host.

    This repo's macOS CI/dev machines have no CUDA device, so the real probe
    must report ``available=False`` here -- a ``True`` result on this host
    would be a false positive the ADR-0002 fail-fast policy exists to prevent.
    """
    pytest.importorskip("torch")

    capability = probe_cuda_capability()

    assert capability.available is False
    assert capability.reason
