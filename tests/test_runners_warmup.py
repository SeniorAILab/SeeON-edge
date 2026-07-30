"""Warmup readiness gate (ADR-0002, acceptance 3 structural).

warmup_to_ready runs the runner warmup, syncs CUDA on a cuda device, gates
readiness, and propagates warmup failures for the bootstrap to fail-fast on.
"""

from __future__ import annotations

import pytest

from edge.runners.warmup import WarmupResult, warmup_runner, warmup_to_ready


class _Runner:
    def __init__(self, fail=False):
        self.warmed = 0
        self._fail = fail

    def warmup(self):
        self.warmed += 1
        if self._fail:
            raise RuntimeError("warmup forward failed")


class _NoWarmup:
    pass


def test_warmup_runner_backward_compatible_no_hook():
    r = _NoWarmup()
    assert warmup_runner(r) is r  # legacy: returns runner unchanged


def test_warmup_to_ready_calls_warmup_and_is_ready_on_cpu():
    r = _Runner()
    result = warmup_to_ready(r, device="cpu")
    assert isinstance(result, WarmupResult)
    assert r.warmed == 1
    assert result.ready is True
    assert result.synchronized is False  # no cuda sync on cpu


def test_warmup_to_ready_synchronizes_on_cuda_device():
    r = _Runner()
    calls = []
    result = warmup_to_ready(r, device="cuda", synchronize=lambda: calls.append("sync"))
    assert calls == ["sync"]
    assert result.synchronized is True
    assert result.ready is True


def test_warmup_to_ready_syncs_for_indexed_cuda_device():
    r = _Runner()
    calls = []
    warmup_to_ready(r, device="cuda:0", synchronize=lambda: calls.append("sync"))
    assert calls == ["sync"]


def test_warmup_to_ready_propagates_warmup_failure_for_failfast():
    r = _Runner(fail=True)
    called = []
    with pytest.raises(RuntimeError, match="warmup forward failed"):
        warmup_to_ready(r, device="cuda", synchronize=lambda: called.append("sync"))
    # warmup failed before sync; readiness never granted, error propagates to bootstrap
    assert called == []
