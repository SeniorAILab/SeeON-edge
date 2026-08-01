from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import BusinessEvent, DecisionInput


@runtime_checkable
class Decider(Protocol):
    """Update one domain's temporal state from image-free numeric input."""

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]: ...


__all__ = ["Decider"]
