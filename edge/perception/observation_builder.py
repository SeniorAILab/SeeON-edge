"""Builders for the FrameObservation contract."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from contracts.observation import BoundingBox, DetectionLabel, DetectionResult, FrameObservation

RawBox = tuple[int, int, int, int, float]
Pose = tuple[tuple[int, int, float], ...]


def observation_from_detection_result(result: DetectionResult) -> FrameObservation:
    """Return a FrameObservation with the same payload as an existing result."""
    return FrameObservation.from_detection_result(result)


def build_frame_observation(
    *,
    detections: DetectionResult | None = None,
    boxes: Iterable[BoundingBox] | None = None,
    labels: Iterable[DetectionLabel] | None = None,
    raw_boxes: Iterable[RawBox] | None = None,
    poses: Iterable[Sequence[tuple[int, int, float]]] | None = None,
    bed_boxes: Iterable[BoundingBox] | None = None,
    bed_exit_statuses: Iterable[object] | None = None,
) -> FrameObservation:
    """Build a FrameObservation from runner-shaped outputs.

    ``detections`` is the compatibility path for existing DetectionResult producers.
    ``raw_boxes`` accepts runner boxes shaped as ``(x1, y1, x2, y2, confidence)``.
    Explicit keyword payloads override the corresponding payload from ``detections``.
    """
    base = (
        FrameObservation.from_detection_result(detections)
        if detections is not None
        else FrameObservation()
    )
    observation_boxes = tuple(boxes) if boxes is not None else base.boxes
    if raw_boxes is not None:
        observation_boxes = tuple(
            BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            for x1, y1, x2, y2, confidence in raw_boxes
        )
    return FrameObservation(
        detections=(
            observation_boxes,
            tuple(labels) if labels is not None else base.labels,
        ),
        poses=tuple(tuple(pose) for pose in poses) if poses is not None else base.keypoints,
        regions=(
            tuple(bed_boxes) if bed_boxes is not None else base.bed_boxes,
            tuple(bed_exit_statuses) if bed_exit_statuses is not None else base.bed_exit_statuses,
        ),
    )
