from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable

from contracts.runner import RunnerProtocol, RunnerResult
from worker.types import FramePacket

_ServingOption: TypeAlias = str | int | float | bool | None


@runtime_checkable
class ServingClient(Protocol):
    def create(self, task: str, **kwargs: _ServingOption) -> RunnerProtocol: ...


@runtime_checkable
class BatchServingClient(ServingClient, Protocol):
    """Typed deferred seam for future cross-camera batched inference."""

    def infer_batch(
        self,
        task: str,
        frames: Sequence[FramePacket],
        **kwargs: _ServingOption,
    ) -> tuple[RunnerResult, ...]: ...


__all__ = ["BatchServingClient", "ServingClient"]
