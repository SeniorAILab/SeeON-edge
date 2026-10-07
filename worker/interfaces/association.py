from __future__ import annotations

from dataclasses import dataclass

from worker.types.perception_frame import PerceptionFrameIdentity

NVDCF_STRATEGY = "nvdcf"


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    pose_row: int | None


@dataclass(frozen=True, slots=True)
class AssociationObservation:
    identity: PerceptionFrameIdentity
    strategy: str
    tracks: tuple[TrackedObject, ...]
    live_track_ids: tuple[int, ...]
    unmatched_tracks: int
    rows_available: int
