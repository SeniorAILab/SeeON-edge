from __future__ import annotations

from dataclasses import dataclass

from contracts.observation import BedRegionDebugSnapshot, FrameObservation
from edge.perception.scene_state import SceneState


@dataclass(frozen=True, slots=True)
class DomainInput:
    """Fully prepared per-frame input for domain interpretation."""

    observation: FrameObservation
    frame_width: int
    frame_height: int
    live_track_ids: tuple[int, ...]
    time_sec: float | None
    frame_index: int
    bed_region: BedRegionDebugSnapshot


def build_domain_input(
    observation: FrameObservation,
    *,
    frame_width: int,
    frame_height: int,
    live_track_ids: tuple[int, ...],
    time_sec: float | None,
    frame_index: int,
    scene_state: SceneState,
    bed_scheduled: bool,
    bed_interval: int,
) -> DomainInput:
    resolved_observation, bed_region = scene_state.resolve_bed_regions(
        observation,
        frame_index=frame_index,
        bed_scheduled=bed_scheduled,
        bed_interval=bed_interval,
    )
    return DomainInput(
        observation=resolved_observation,
        frame_width=frame_width,
        frame_height=frame_height,
        live_track_ids=live_track_ids,
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=bed_region,
    )
