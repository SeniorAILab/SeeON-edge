from __future__ import annotations

from contracts.observation import BoundingBox, FrameObservation
from edge.perception.scene_state import SceneState

BED = BoundingBox(0, 0, 10, 10, 0.9)


def _observation(beds: tuple[BoundingBox, ...] = ()) -> FrameObservation:
    return FrameObservation(regions=(beds, ()))


def test_scene_state_carries_bed_roi_until_frame_index_ttl() -> None:
    state = SceneState("cam-1")
    observation, debug = state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "fresh"

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=60, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "cached"
    assert debug.age_frames == 60

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == ()
    assert debug.source == "expired"


def test_scene_state_invalidates_after_two_scheduled_empty_bed_cycles() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    first, first_debug = state.resolve_bed_regions(
        _observation(), frame_index=30, bed_scheduled=True, bed_interval=30
    )
    assert first.bed_boxes == ()
    assert first_debug.source == "empty"
    second, second_debug = state.resolve_bed_regions(
        _observation(), frame_index=60, bed_scheduled=True, bed_interval=30
    )
    assert second.bed_boxes == ()
    assert second_debug.source == "expired"
    carried, carried_debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=False, bed_interval=30
    )
    assert carried.bed_boxes == ()
    assert carried_debug.source == "empty"


def test_scene_state_resets_on_frame_index_discontinuity() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    state.resolve_bed_regions(_observation(), frame_index=1, bed_scheduled=False, bed_interval=30)
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == ()
    assert debug.reset_reason == "frame_index_discontinuity"
    assert state.bed_region_counters.reset == 1


def test_cached_roi_freshness_counters_are_observable() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=1
    )
    state.resolve_bed_regions(_observation(), frame_index=1, bed_scheduled=False, bed_interval=1)
    state.resolve_bed_regions(_observation(), frame_index=3, bed_scheduled=False, bed_interval=1)
    state.resolve_bed_regions(_observation(), frame_index=0, bed_scheduled=False, bed_interval=1)
    assert state.bed_region_counters.snapshot()["fresh"] == 1
    assert state.bed_region_counters.snapshot()["cached"] == 1
    assert state.bed_region_counters.snapshot()["expired"] == 1
    assert state.bed_region_counters.snapshot()["reset"] == 1
