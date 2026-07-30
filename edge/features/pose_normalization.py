from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_N_KEYPOINTS = 17
_KPT_DIMS = 3


def normalize_person_keypoints(
    pose_detections: tuple,
    frame_w: int,
    frame_h: int,
    conf_threshold: float,
) -> NDArray[np.float32]:
    """Convert raw pose detections to a normalised float32 array of shape (17, 3)."""
    out = np.zeros((_N_KEYPOINTS, _KPT_DIMS), dtype=np.float32)

    person = pose_detections[0] if len(pose_detections) > 0 else None
    if person is None:
        return out

    for i, (x_int, y_int, conf) in enumerate(person):
        if i >= _N_KEYPOINTS:
            break
        if conf < conf_threshold:
            # Zero x, y, AND conf so `conf > 0` downstream excludes this keypoint.
            out[i] = (0.0, 0.0, 0.0)
        else:
            out[i] = (
                float(x_int) / frame_w,
                float(y_int) / frame_h,
                float(conf),
            )

    return out
