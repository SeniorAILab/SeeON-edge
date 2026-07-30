"""Fail-fast GPU device policy (ADR-0002).

Covers `require_gpu_device` (fail-fast, no silent CPU fallback) and `probe_cuda`
(precise diagnosis incl. the empty-arch-list / missing-sm_120 root cause), while
asserting the legacy `select_device` stays backward-compatible.
"""

from __future__ import annotations

import pytest

from edge.runners import device as dev


def _probe(available, reason="x", device_count=0, arch_list=()):
    return dev.CudaProbe(
        available=available, reason=reason, device_count=device_count, arch_list=tuple(arch_list)
    )


def test_require_gpu_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(dev, "probe_cuda", lambda: _probe(True, "cuda available", 1, ("sm_120",)))
    assert dev.require_gpu_device() == "cuda"


def test_require_gpu_device_fails_fast_when_unavailable(monkeypatch):
    monkeypatch.setattr(dev, "probe_cuda", lambda: _probe(False, "no compiled arch kernels"))
    with pytest.raises(dev.GpuUnavailableError) as exc:
        dev.require_gpu_device()
    # never silently returns "cpu"; the reason is surfaced, not masked
    assert "no compiled arch kernels" in str(exc.value)


def test_require_gpu_device_honors_explicit_cpu_optin(monkeypatch):
    # explicit operator opt-in must NOT fail fast even when CUDA is unusable
    monkeypatch.setattr(dev, "probe_cuda", lambda: _probe(False, "unusable"))
    assert dev.require_gpu_device("cpu") == "cpu"
    assert dev.require_gpu_device("opencv") == "opencv"


def test_select_device_backward_compatible(monkeypatch):
    monkeypatch.setattr(dev, "probe_cuda", lambda: _probe(False, "unusable"))
    assert dev.select_device() == "cpu"  # legacy conservative degrade preserved
    assert dev.select_device("cuda") == "cuda"  # explicit wins
    monkeypatch.setattr(dev, "probe_cuda", lambda: _probe(True, "ok", 1, ("sm_120",)))
    assert dev.select_device() == "cuda"


def test_probe_cuda_diagnoses_empty_arch_list_root_cause(monkeypatch):
    """The ADR-0002 root cause: GPU visible but torch build has no arch kernels."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])

    probe = dev.probe_cuda()
    assert probe.available is False
    assert probe.device_count == 1
    assert probe.arch_list == ()
    assert "arch" in probe.reason.lower()
    assert "ADR-0002" in probe.reason
