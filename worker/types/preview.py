from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FallPreviewStatus = Literal["normal", "suspected"]


@dataclass(frozen=True, slots=True)
class OverlaySelection:
    person: bool = True
    bed: bool = True


@dataclass(frozen=True, slots=True)
class FallPreviewState:
    track_id: int
    status: FallPreviewStatus
    probability: float | None = None


__all__ = ["FallPreviewState", "FallPreviewStatus", "OverlaySelection"]
