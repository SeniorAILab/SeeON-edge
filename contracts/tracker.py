from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.observation import BoundingBox


@runtime_checkable
class TrackerProtocol(Protocol):
    def update(self, boxes: tuple[BoundingBox, ...]) -> tuple[int, ...]: ...

    @property
    def live_ids(self) -> frozenset[int]: ...


__all__ = ["TrackerProtocol"]
