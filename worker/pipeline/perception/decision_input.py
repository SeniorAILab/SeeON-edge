from __future__ import annotations

from contracts.observation import FrameObservation
from worker.pipeline.perception.scene_state import SceneState
from worker.types import DecisionInput


def build_decision_input(
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
) -> DecisionInput:
    """Resolve scene provenance and return the image-free decision boundary."""
    resolved_observation, bed_region = scene_state.resolve_bed_regions(
        observation,
        frame_index=frame_index,
        bed_scheduled=bed_scheduled,
        bed_interval=bed_interval,
    )
    return DecisionInput(
        observation=resolved_observation,
        frame_width=frame_width,
        frame_height=frame_height,
        live_track_ids=live_track_ids,
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=bed_region,
    )


__all__ = ["build_decision_input"]
