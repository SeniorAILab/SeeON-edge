from __future__ import annotations

from typing import Any

import pytest

from worker.adapters.device.nvml.probe import NvmlGpuStatus, probe_nvml_gpu_status


class _FakePynvml:
    def __init__(
        self,
        *,
        init_error: Exception | None = None,
        device_count: int | Exception = 1,
        driver_version: str | Exception = "550.90.07",
        device_name: str | Exception = "Tesla T4",
    ) -> None:
        self._init_error = init_error
        self._device_count = device_count
        self._driver_version = driver_version
        self._device_name = device_name
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        if self._init_error is not None:
            raise self._init_error

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:
        if isinstance(self._device_count, Exception):
            raise self._device_count
        return self._device_count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        del index
        return object()

    def nvmlDeviceGetName(self, handle: object) -> str:
        del handle
        if isinstance(self._device_name, Exception):
            raise self._device_name
        return self._device_name

    def nvmlSystemGetDriverVersion(self) -> str:
        if isinstance(self._driver_version, Exception):
            raise self._driver_version
        return self._driver_version


def test_nvml_gpu_status_true_when_available() -> None:
    fake = _FakePynvml(device_count=1, driver_version="550.90.07", device_name="Tesla T4")

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status == NvmlGpuStatus(
        nvml_available=True,
        reason="NVML reports a usable GPU device",
        driver_version="550.90.07",
        device_name="Tesla T4",
    )
    assert fake.shutdown_called is True


def test_nvml_gpu_status_false_when_pynvml_import_fails() -> None:
    def failing_importer() -> Any:
        raise ImportError("no module named pynvml")

    status = probe_nvml_gpu_status(importer=failing_importer)

    assert status.nvml_available is False
    assert "pynvml import failed" in status.reason
    assert status.driver_version is None
    assert status.device_name is None


def test_nvml_gpu_status_false_when_nvml_init_fails() -> None:
    fake = _FakePynvml(init_error=RuntimeError("NVML Shared Library Not Found"))

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status.nvml_available is False
    assert "nvmlInit failed" in status.reason
    assert "NVML Shared Library Not Found" in status.reason
    assert fake.shutdown_called is False


def test_nvml_gpu_status_false_when_no_devices_visible() -> None:
    fake = _FakePynvml(device_count=0)

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status.nvml_available is False
    assert "no GPU devices are visible" in status.reason
    assert status.device_name is None
    assert status.driver_version == "550.90.07"
    assert fake.shutdown_called is True


def test_nvml_gpu_status_false_when_device_count_query_raises() -> None:
    fake = _FakePynvml(device_count=RuntimeError("driver error"))

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status.nvml_available is False
    assert "nvmlDeviceGetCount failed" in status.reason
    assert fake.shutdown_called is True


def test_nvml_gpu_status_available_but_device_name_none_when_name_query_fails() -> None:
    fake = _FakePynvml(device_count=1, device_name=RuntimeError("name query failed"))

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status.nvml_available is True
    assert status.device_name is None
    assert status.driver_version == "550.90.07"


def test_nvml_gpu_status_driver_version_none_when_driver_version_query_fails() -> None:
    fake = _FakePynvml(device_count=1, driver_version=RuntimeError("driver query failed"))

    status = probe_nvml_gpu_status(importer=lambda: fake)

    assert status.nvml_available is True
    assert status.driver_version is None
    assert status.device_name == "Tesla T4"


def test_probe_nvml_gpu_status_against_real_pynvml_always_states_a_reason() -> None:
    pytest.importorskip("pynvml")

    status = probe_nvml_gpu_status()

    assert status.reason
