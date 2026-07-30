from __future__ import annotations

import sys
from types import SimpleNamespace

from edge.runners.device import select_device


class _CudaProbe:
    def __init__(self, available: bool | None = None, error: Exception | None = None) -> None:
        self._available = available
        self._error = error

    def is_available(self) -> bool:
        if self._error is not None:
            raise self._error
        return bool(self._available)


def test_select_device_returns_cuda_when_cuda_available(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_CudaProbe(True)))

    assert select_device() == "cuda"


def test_select_device_returns_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_CudaProbe(False)))

    assert select_device() == "cpu"


def test_select_device_returns_cpu_when_detection_raises(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=_CudaProbe(error=RuntimeError("cuda probe failed"))),
    )

    assert select_device() == "cpu"


def test_select_device_respects_explicit_override(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_CudaProbe(False)))

    assert select_device("cuda") == "cuda"
