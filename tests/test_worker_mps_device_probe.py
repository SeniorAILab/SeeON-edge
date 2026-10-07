from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worker.adapters.device.mps.probe import MpsCapability, probe_mps_capability


class _FakeMpsBackend:
    def __init__(
        self,
        *,
        is_built: Callable[[], bool] | Exception = True,
        is_available: Callable[[], bool] | Exception = False,
    ) -> None:
        self._is_built = is_built
        self._is_available = is_available

    def is_built(self) -> bool:
        if isinstance(self._is_built, Exception):
            raise self._is_built
        return self._is_built

    def is_available(self) -> bool:
        if isinstance(self._is_available, Exception):
            raise self._is_available
        return self._is_available


class _FakeBackends:
    def __init__(self, mps: _FakeMpsBackend) -> None:
        self.mps = mps


class _FakeTorch:
    def __init__(self, backends: _FakeBackends) -> None:
        self.backends = backends


def test_mps_capability_true_when_available() -> None:
    fake_torch = _FakeTorch(_FakeBackends(_FakeMpsBackend(is_built=True, is_available=True)))

    capability = probe_mps_capability(importer=lambda: fake_torch)

    assert capability == MpsCapability(available=True, reason="mps available", is_built=True)


def test_mps_capability_false_when_torch_import_fails() -> None:
    def failing_importer() -> Any:
        raise ImportError("no module named torch")

    capability = probe_mps_capability(importer=failing_importer)

    assert capability.available is False
    assert "torch import failed" in capability.reason
    assert capability.is_built is False


def test_mps_capability_false_when_not_built() -> None:
    fake_torch = _FakeTorch(_FakeBackends(_FakeMpsBackend(is_built=False, is_available=False)))

    capability = probe_mps_capability(importer=lambda: fake_torch)

    assert capability.available is False
    assert capability.is_built is False
    assert "is_built() is False" in capability.reason


def test_mps_capability_false_when_built_but_no_device_available() -> None:
    fake_torch = _FakeTorch(_FakeBackends(_FakeMpsBackend(is_built=True, is_available=False)))

    capability = probe_mps_capability(importer=lambda: fake_torch)

    assert capability.available is False
    assert capability.is_built is True
    assert "no usable Metal device" in capability.reason


def test_mps_capability_false_when_is_available_raises() -> None:
    fake_torch = _FakeTorch(
        _FakeBackends(_FakeMpsBackend(is_built=True, is_available=RuntimeError("backend error")))
    )

    capability = probe_mps_capability(importer=lambda: fake_torch)

    assert capability.available is False
    assert "torch.backends.mps.is_available() raised" in capability.reason
    assert capability.is_built is True


def test_mps_capability_defaults_is_built_false_when_is_built_raises() -> None:
    fake_torch = _FakeTorch(
        _FakeBackends(_FakeMpsBackend(is_built=RuntimeError("no build flag"), is_available=False))
    )

    capability = probe_mps_capability(importer=lambda: fake_torch)

    assert capability.is_built is False
    assert capability.available is False
