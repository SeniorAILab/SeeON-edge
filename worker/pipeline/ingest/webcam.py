from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np

from contracts.frame import Frame


class CameraSource:
    def __init__(self, device_index: int, max_failures: int = 30) -> None:
        self._device_index = device_index
        self._max_failures = max_failures

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self._device_index)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            t0: float | None = None
            frame_index = 0
            consecutive_failures = 0
            while True:
                read_ok, frame_bgr = capture.read()
                if not read_ok:
                    consecutive_failures += 1
                    if consecutive_failures >= self._max_failures:
                        break
                    continue
                consecutive_failures = 0
                now = time.monotonic()
                if t0 is None:
                    t0 = now
                image = np.asarray(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    dtype=np.uint8,
                )
                yield Frame(
                    index=frame_index,
                    time_sec=round(now - t0, 3),
                    image=image,
                )
                frame_index += 1
        finally:
            capture.release()


__all__ = ["CameraSource"]
