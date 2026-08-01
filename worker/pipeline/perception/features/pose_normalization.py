from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

Keypoint: TypeAlias = tuple[int, int, float]
PersonPose: TypeAlias = Sequence[Keypoint]
PoseDetections: TypeAlias = Sequence[PersonPose]

_N_KEYPOINTS: Final = 17
_KEYPOINT_DIMS: Final = 3


def normalize_person_keypoints(
    pose_detections: PoseDetections,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
) -> NDArray[np.float32]:
    """Normalize the first detected COCO-17 pose to ``float32[17, 3]``."""
    normalized = np.zeros((_N_KEYPOINTS, _KEYPOINT_DIMS), dtype=np.float32)
    if not pose_detections:
        return normalized

    for index, (x_coordinate, y_coordinate, confidence) in enumerate(pose_detections[0]):
        if index >= _N_KEYPOINTS:
            break
        if confidence < confidence_threshold:
            normalized[index] = (0.0, 0.0, 0.0)
            continue
        normalized[index] = (
            float(x_coordinate) / frame_width,
            float(y_coordinate) / frame_height,
            float(confidence),
        )
    return normalized


__all__ = ["normalize_person_keypoints"]
