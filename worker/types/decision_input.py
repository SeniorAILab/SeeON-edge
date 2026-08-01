from __future__ import annotations

from dataclasses import dataclass, field

from contracts.observation import BedRegionDebugSnapshot, FrameObservation


@dataclass(frozen=True, slots=True)
class DecisionInput:
    observation: FrameObservation = field(hash=False)
    frame_width: int
    frame_height: int
    live_track_ids: tuple[int, ...]
    time_sec: float | None
    frame_index: int
    bed_region: BedRegionDebugSnapshot


__all__ = ["DecisionInput"]
