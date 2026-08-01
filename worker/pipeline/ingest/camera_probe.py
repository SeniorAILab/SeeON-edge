from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraInfo:
    index: int
    thumbnail: NDArray[np.uint8]


def probe_cameras(max_index: int = 5) -> list[CameraInfo]:
    found: list[CameraInfo] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        read_ok, frame_bgr = capture.read()
        capture.release()
        if read_ok and frame_bgr is not None:
            thumbnail = np.asarray(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                dtype=np.uint8,
            )
            found.append(CameraInfo(index=index, thumbnail=thumbnail))
    return found


__all__ = ["CameraInfo", "probe_cameras"]
