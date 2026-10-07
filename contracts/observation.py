from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

FALL_LABEL_TEXT: Final = "FALL"
NORMAL_LABEL_TEXT: Final = "NORMAL"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    polygon: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class DetectionLabel:
    text: str
    confidence: float
    is_fall: bool


Detections = tuple[tuple[BoundingBox, ...], tuple[DetectionLabel, ...]]
Regions = tuple[tuple[BoundingBox, ...], tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    boxes: tuple[BoundingBox, ...] = field(default_factory=tuple)
    labels: tuple[DetectionLabel, ...] = field(default_factory=tuple)
    keypoints: tuple[tuple[tuple[int, int, float], ...], ...] = field(default_factory=tuple)
    bed_boxes: tuple[BoundingBox, ...] = field(default_factory=tuple)
    bed_exit_statuses: tuple[object, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class FrameObservation:
    detections: Detections = field(default_factory=lambda: ((), ()))
    poses: tuple[tuple[tuple[int, int, float], ...], ...] = field(default_factory=tuple)
    regions: Regions = field(default_factory=lambda: ((), ()))
    track_ids: tuple[int | None, ...] = field(default_factory=tuple)

    @property
    def boxes(self) -> tuple[BoundingBox, ...]:
        return self.detections[0]

    @property
    def labels(self) -> tuple[DetectionLabel, ...]:
        return self.detections[1]

    @property
    def keypoints(self) -> tuple[tuple[tuple[int, int, float], ...], ...]:
        return self.poses

    @property
    def bed_boxes(self) -> tuple[BoundingBox, ...]:
        return self.regions[0]

    @property
    def bed_exit_statuses(self) -> tuple[object, ...]:
        return self.regions[1]

    @classmethod
    def from_detection_result(cls, result: DetectionResult) -> FrameObservation:
        return cls(
            detections=(result.boxes, result.labels),
            poses=result.keypoints,
            regions=(result.bed_boxes, result.bed_exit_statuses),
        )

    def to_detection_result(self) -> DetectionResult:
        boxes, labels = self.detections
        bed_boxes, bed_exit_statuses = self.regions
        return DetectionResult(
            boxes=boxes,
            labels=labels,
            keypoints=self.poses,
            bed_boxes=bed_boxes,
            bed_exit_statuses=bed_exit_statuses,
        )


class BedRegionCacheState(StrEnum):
    FRESH = "fresh"
    CACHED = "cached"
    EMPTY = "empty"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class BedRegionDebugSnapshot:
    source: BedRegionCacheState
    empty_cycles: int = 0
