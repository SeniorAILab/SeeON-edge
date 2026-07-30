"""Per-camera scene-state aggregation for perception."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)


@dataclass(slots=True)
class BedRegionCacheCounters:
    fresh: int = 0
    cached: int = 0
    expired: int = 0
    reset: int = 0
    scheduled_empty: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "fresh": self.fresh,
            "cached": self.cached,
            "expired": self.expired,
            "reset": self.reset,
            "scheduled_empty": self.scheduled_empty,
        }


@dataclass(slots=True)
class SceneState:
    """Cache the latest observation, track ids, and bed regions for one camera."""

    camera_id: str
    latest_observation: FrameObservation | None = None
    track_ids: tuple[int, ...] = field(default_factory=tuple)
    bed_regions: tuple[BoundingBox, ...] = field(default_factory=tuple)
    last_bed_frame_index: int | None = None
    last_processed_frame_index: int | None = None
    scheduled_empty_bed_cycles: int = 0
    bed_region_freshness: BedRegionCacheState = BedRegionCacheState.EMPTY
    bed_region_counters: BedRegionCacheCounters = field(default_factory=BedRegionCacheCounters)

    def reset_bed_cache(self, reason: str) -> None:
        """Clear cached bed regions and record the reset reason."""
        self.bed_regions = ()
        self.last_bed_frame_index = None
        self.scheduled_empty_bed_cycles = 0
        self.bed_region_freshness = BedRegionCacheState.EMPTY
        self.bed_region_counters.reset += 1

    def reset_for_new_source(self, reason: str = "source_restart") -> None:
        """Clear frame-continuity and bed ROI state at a source iterator boundary."""
        self.reset_bed_cache(reason)
        self.last_processed_frame_index = None
        self.latest_observation = None
        self.track_ids = ()

    def update(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        """Store the newest observation and refresh cached scene fields."""
        self.latest_observation = observation
        self.track_ids = track_ids
        bed_boxes, _statuses = observation.regions
        if bed_boxes:
            self.bed_regions = bed_boxes
        return observation

    def resolve_bed_regions(
        self,
        observation: FrameObservation,
        *,
        frame_index: int,
        bed_scheduled: bool,
        bed_interval: int,
    ) -> tuple[FrameObservation, BedRegionDebugSnapshot]:
        """Resolve fresh, cached, empty, or expired bed ROIs for a frame.

        Fresh scheduled non-empty bed output refreshes the cache. Unscheduled frames may carry
        the latest cache only while it is within ``2 * bed_interval`` and fewer than two
        scheduled-empty bed cycles have been observed. Frame index discontinuity starts a new
        source sequence and clears stale cache before evaluating this frame.
        """
        reset_reason = self._reset_if_discontinuous(frame_index)
        bed_boxes = observation.bed_boxes

        if bed_scheduled and bed_boxes:
            self.bed_regions = bed_boxes
            self.last_bed_frame_index = frame_index
            self.scheduled_empty_bed_cycles = 0
            self.bed_region_freshness = BedRegionCacheState.FRESH
            self.bed_region_counters.fresh += 1
            self._mark_processed(frame_index, observation)
            return observation, BedRegionDebugSnapshot(
                source=BedRegionCacheState.FRESH,
                age_frames=0,
                empty_cycles=0,
                reset_reason=reset_reason,
            )

        if bed_scheduled:
            self.scheduled_empty_bed_cycles += 1
            self.bed_region_counters.scheduled_empty += 1
            if self.scheduled_empty_bed_cycles >= 2:
                self.bed_regions = ()
                self.last_bed_frame_index = None
                self.bed_region_freshness = BedRegionCacheState.EXPIRED
                self.bed_region_counters.expired += 1
                snapshot = BedRegionDebugSnapshot(
                    source=BedRegionCacheState.EXPIRED,
                    age_frames=None,
                    empty_cycles=self.scheduled_empty_bed_cycles,
                    reset_reason=reset_reason,
                )
                resolved = _replace_bed_boxes(observation, ())
                self._mark_processed(frame_index, resolved)
                return resolved, snapshot
            self.bed_region_freshness = BedRegionCacheState.EMPTY
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.EMPTY,
                age_frames=self._age(frame_index),
                empty_cycles=self.scheduled_empty_bed_cycles,
                reset_reason=reset_reason,
            )
            self._mark_processed(frame_index, observation)
            return observation, snapshot

        cached = self._cached_boxes_if_fresh(frame_index, bed_interval)
        if cached:
            resolved = _replace_bed_boxes(observation, cached)
            self.bed_region_freshness = BedRegionCacheState.CACHED
            self.bed_region_counters.cached += 1
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.CACHED,
                age_frames=self._age(frame_index),
                empty_cycles=self.scheduled_empty_bed_cycles,
                reset_reason=reset_reason,
            )
            self._mark_processed(frame_index, resolved)
            return resolved, snapshot

        source: BedRegionCacheState = (
            BedRegionCacheState.EXPIRED
            if self.bed_regions or self.last_bed_frame_index is not None
            else BedRegionCacheState.EMPTY
        )
        if source == BedRegionCacheState.EXPIRED:
            self.bed_regions = ()
            self.last_bed_frame_index = None
            self.bed_region_freshness = BedRegionCacheState.EXPIRED
            self.bed_region_counters.expired += 1
        else:
            self.bed_region_freshness = BedRegionCacheState.EMPTY
        resolved = _replace_bed_boxes(observation, ())
        snapshot = BedRegionDebugSnapshot(
            source=source,
            age_frames=None,
            empty_cycles=self.scheduled_empty_bed_cycles,
            reset_reason=reset_reason,
        )
        self._mark_processed(frame_index, resolved)
        return resolved, snapshot

    def _reset_if_discontinuous(self, frame_index: int) -> str | None:
        if self.last_processed_frame_index is None:
            return None
        if frame_index > self.last_processed_frame_index:
            return None
        self.bed_regions = ()
        self.last_bed_frame_index = None
        self.scheduled_empty_bed_cycles = 0
        self.bed_region_freshness = BedRegionCacheState.EMPTY
        self.bed_region_counters.reset += 1
        return "frame_index_discontinuity"

    def _cached_boxes_if_fresh(
        self,
        frame_index: int,
        bed_interval: int,
    ) -> tuple[BoundingBox, ...]:
        if not self.bed_regions or self.last_bed_frame_index is None:
            return ()
        if self.scheduled_empty_bed_cycles >= 2:
            return ()
        age_frames = frame_index - self.last_bed_frame_index
        if age_frames < 0:
            return ()
        if age_frames <= 2 * max(1, bed_interval):
            return self.bed_regions
        return ()

    def _age(self, frame_index: int) -> int | None:
        if self.last_bed_frame_index is None:
            return None
        return max(0, frame_index - self.last_bed_frame_index)

    def _mark_processed(self, frame_index: int, observation: FrameObservation) -> None:
        self.last_processed_frame_index = frame_index
        self.update(observation)


def _replace_bed_boxes(
    observation: FrameObservation,
    bed_boxes: tuple[BoundingBox, ...],
) -> FrameObservation:
    return replace(observation, regions=(bed_boxes, observation.bed_exit_statuses))
