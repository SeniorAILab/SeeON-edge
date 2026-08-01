from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from contracts.observation import FrameObservation
from worker.pipeline.analytics.merge import authoritative_boxes, merge_module_results
from worker.pipeline.analytics.models import NamedExtractor, ensure_unique_module_names
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import (
    GreedyIouTracker,
    SceneState,
    build_decision_input,
    build_frame_observation,
)
from worker.types import DecisionInput, FramePacket, ModuleResult


@dataclass(frozen=True, slots=True)
class CompositeResult:
    module_results: tuple[ModuleResult, ...]
    observation: FrameObservation
    decision_input: DecisionInput


class CompositeExtractor:
    """Own one camera's analytics state while reusing shared named extractors."""

    def __init__(
        self,
        *,
        extractors: Sequence[NamedExtractor],
        scheduler: Scheduler,
        tracker: GreedyIouTracker,
        scene_state: SceneState,
    ) -> None:
        frozen_extractors = tuple(extractors)
        ensure_unique_module_names(
            tuple(extractor.module_name for extractor in frozen_extractors)
        )
        self.extractors: tuple[NamedExtractor, ...] = frozen_extractors
        self.scheduler: Scheduler = scheduler
        self.tracker: GreedyIouTracker = tracker
        self.scene_state: SceneState = scene_state
        self._extractors_by_name: dict[str, NamedExtractor] = {
            extractor.module_name: extractor for extractor in frozen_extractors
        }

    def process(self, packet: FramePacket) -> CompositeResult:
        """Run due modules and emit one tracked, image-free decision input."""
        scheduled_names = self.scheduler.tasks_for_frame(packet.frame.index)
        module_results = tuple(
            extractor.extract(packet)
            for name in scheduled_names
            if (extractor := self._extractors_by_name.get(name)) is not None
        )
        merged = merge_module_results(module_results)
        track_ids = self.tracker.update(authoritative_boxes(merged))
        observation = build_frame_observation(
            detections=merged.detections,
            raw_boxes=merged.raw_boxes,
            poses=merged.poses,
            bed_boxes=merged.bed_boxes,
            track_ids=track_ids,
        )
        decision_input = build_decision_input(
            observation,
            frame_width=packet.width,
            frame_height=packet.height,
            live_track_ids=tuple(sorted(self.tracker.live_ids)),
            time_sec=packet.frame.time_sec,
            frame_index=packet.frame.index,
            scene_state=self.scene_state,
            bed_scheduled="bed" in scheduled_names,
            bed_interval=self.scheduler.task_intervals.get("bed", 30),
        )
        final_observation = decision_input.observation
        _ = self.scene_state.update(final_observation, track_ids=track_ids)
        return CompositeResult(
            module_results=module_results,
            observation=final_observation,
            decision_input=decision_input,
        )
__all__ = ["CompositeExtractor", "CompositeResult"]
