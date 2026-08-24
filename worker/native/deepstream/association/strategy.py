"""``AssociationStrategy`` port: the seam every native strategy implements.

The native `GstBaseTransform` calls exactly one strategy per camera through
this narrow surface. `AssociationOutcome` carries only what
`AssociationResult` (`worker/types/perception_frame.py`) needs: durable track
ids in cue order, plus the `live_ids` snapshot the fall/bed-exit domains read
through `DecisionInput.live_track_ids`. Bed masks never reach this port --
callers pass person-box cues only (Task 4 guardrail: bed regions are scene
context, never identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contracts.observation import BoundingBox


@dataclass(frozen=True, slots=True)
class AssociationOutcome:
    """One frame's association result: durable ids in incoming cue order."""

    track_ids: tuple[int, ...]
    live_ids: frozenset[int]


@runtime_checkable
class AssociationStrategy(Protocol):
    """Stateful per-camera person-identity assignment across frames."""

    #: Explicit version identity. Config/registry proves only one strategy's
    #: identity is active at cutover; never inferred from the class name.
    identity: str

    @property
    def live_ids(self) -> frozenset[int]:
        """Every track id this strategy currently considers live."""
        ...

    def observe(self, boxes: tuple[BoundingBox, ...]) -> AssociationOutcome:
        """Apply one actual inference observation, including an empty one."""
        ...

    def coast(self) -> None:
        """Preserve every track when this frame carried no inference result."""
        ...

    def reset(self) -> None:
        """Drop all track state. Called on reconnect / stream-epoch rollover.

        A fresh boot id or a rolled stream epoch must never let the next
        frame observe a track id minted before the reset.
        """
        ...


__all__ = ["AssociationOutcome", "AssociationStrategy"]
