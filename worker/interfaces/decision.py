from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


@runtime_checkable
class Decider(Protocol):
    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]: ...


@runtime_checkable
class TraceSnapshotProvider(Decider, Protocol):
    @property
    def last_trace_snapshots(self) -> tuple[DecisionTraceSnapshot, ...]: ...


@runtime_checkable
class FreshnessProvider(TraceSnapshotProvider, Protocol):
    @property
    def last_update_evaluated(self) -> bool: ...


@runtime_checkable
class ShadowTraceProvider(TraceSnapshotProvider, Protocol):
    @property
    def last_shadow_trace_count(self) -> int: ...


__all__ = [
    "Decider",
    "FreshnessProvider",
    "ShadowTraceProvider",
    "TraceSnapshotProvider",
]
