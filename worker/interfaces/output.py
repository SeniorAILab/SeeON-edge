from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import BusinessEvent


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: BusinessEvent) -> None: ...


__all__ = ["EventSink"]
