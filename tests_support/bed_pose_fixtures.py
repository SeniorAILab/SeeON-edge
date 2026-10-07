from __future__ import annotations

from worker.types import BedPoseFeatures, FrameBedPoseFeatures

_LYING_HIP_DEPTH = 0.257
_STANDING_HIP_DEPTH = -0.289


def lying_in_bed(track_id: int, bed_id: int | None = 0) -> BedPoseFeatures:
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=1.0,
        lower_in_frac=1.0,
        keypoint_in_frac=1.0,
        hip_depth=_LYING_HIP_DEPTH,
        torso_angle=1.4,
        centroid_displacement=0.0,
        hip_x_rel=0.5,
        hip_y_rel=0.5,
        observability=0.9,
        bed_polygon_valid=True,
    )


def standing(track_id: int, bed_id: int | None = 0) -> BedPoseFeatures:
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=1.0,
        lower_in_frac=1.0,
        keypoint_in_frac=1.0,
        hip_depth=_STANDING_HIP_DEPTH,
        torso_angle=1.5,
        centroid_displacement=0.0,
        hip_x_rel=0.5,
        hip_y_rel=0.2,
        observability=0.9,
        bed_polygon_valid=True,
    )


def frame_pose_features(*features: BedPoseFeatures) -> FrameBedPoseFeatures:
    return FrameBedPoseFeatures(items=tuple(features))


__all__ = ["frame_pose_features", "lying_in_bed", "standing"]
