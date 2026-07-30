from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraInfo:
    index: int
    thumbnail: NDArray[np.uint8]  # RGB, HWC


def probe_cameras(max_index: int = 5) -> list[CameraInfo]:
    """Probe OpenCV device indices 0..(max_index-1) for live cameras.

    Returns a CameraInfo for each index that opens AND yields at least one
    frame.  The thumbnail is the first captured frame converted to RGB.
    Pure OpenCV — no Streamlit dependency (unit-testable with monkeypatched
    cv2.VideoCapture, same 계약(contract) layer as CameraSource).
    """
    found: list[CameraInfo] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok and frame_bgr is not None:
            thumbnail = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            found.append(CameraInfo(index=idx, thumbnail=thumbnail))
    return found
