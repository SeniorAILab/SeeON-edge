from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TypedDict

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)


class BedRegionCacheCounterSnapshot(TypedDict):
    fresh: int
    cached: int
    expired: int
    reset: int
    scheduled_empty: int


@dataclass(slots=True)
class BedRegionCacheCounters:
    fresh: int = 0
    cached: int = 0
    expired: int = 0
    reset: int = 0
    scheduled_empty: int = 0

    def snapshot(self) -> BedRegionCacheCounterSnapshot:
        return {
            "fresh": self.fresh,
            "cached": self.cached,
            "expired": self.expired,
            "reset": self.reset,
            "scheduled_empty": self.scheduled_empty,
        }


@dataclass(slots=True)
class SceneState:
    camera_id: str
    latest_observation: FrameObservation | None = None
    track_ids: tuple[int, ...] = field(default_factory=tuple)
    scheduled_empty_bed_cycles: int = 0
    bed_region_freshness: BedRegionCacheState = BedRegionCacheState.EMPTY
    bed_region_counters: BedRegionCacheCounters = field(default_factory=BedRegionCacheCounters)
    persisted_bed_regions: tuple[BoundingBox, ...] = field(default_factory=tuple)
    bed_zone_image_width: int | None = None
    bed_zone_image_height: int | None = None

    def reset_bed_cache(self, _reason: str) -> None:
        ...

    def reset_for_new_source(self, reason: str = "source_restart") -> None:
        self.reset_bed_cache(reason)
        self.latest_observation = None
        self.track_ids = ()

    def update(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        return self.observe(observation, track_ids=track_ids)

    def observe(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        self.latest_observation = observation
        self.track_ids = track_ids
        return observation

    def coast(self) -> FrameObservation | None:
        return self.latest_observation

    def resolve_bed_regions(
        self,
        observation: FrameObservation,
        *,
        frame_index: int,
        bed_scheduled: bool,
        bed_interval: int,
    ) -> tuple[FrameObservation, BedRegionDebugSnapshot]:
        _ = frame_index, bed_scheduled, bed_interval
        if self.persisted_bed_regions:
            resolved = _replace_bed_boxes(observation, self.persisted_bed_regions)
            snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH, empty_cycles=0)
            self.bed_region_freshness = snapshot.source
            self._mark_processed(resolved)
            return resolved, snapshot

        resolved = _replace_bed_boxes(observation, ())
        snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY, empty_cycles=0)
        self.bed_region_freshness = snapshot.source
        self._mark_processed(resolved)
        return resolved, snapshot

    @property
    def bed_polygon_source(self) -> str:
        return "persisted" if self.persisted_bed_regions else "none"

    def _mark_processed(self, observation: FrameObservation) -> None:
        _ = self.update(observation)


def _replace_bed_boxes(
    observation: FrameObservation,
    bed_boxes: tuple[BoundingBox, ...],
) -> FrameObservation:
    return replace(observation, regions=(bed_boxes, observation.bed_exit_statuses))


__all__ = ["BedRegionCacheCounters", "SceneState"]
