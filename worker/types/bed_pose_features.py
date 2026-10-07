from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BedPoseFeatures:
    track_id: int
    bed_id: int | None
    torso_in_frac: float
    lower_in_frac: float
    keypoint_in_frac: float
    hip_depth: float
    torso_angle: float
    centroid_displacement: float
    hip_x_rel: float
    hip_y_rel: float
    observability: float
    bed_polygon_valid: bool


@dataclass(frozen=True, slots=True)
class FrameBedPoseFeatures:
    items: tuple[BedPoseFeatures, ...] = ()


EMPTY_FRAME_BED_POSE_FEATURES = FrameBedPoseFeatures()


__all__ = [
    "EMPTY_FRAME_BED_POSE_FEATURES",
    "BedPoseFeatures",
    "FrameBedPoseFeatures",
]
