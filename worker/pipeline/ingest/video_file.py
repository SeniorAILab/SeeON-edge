from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from contracts.frame import Frame


class VideoFileSource:
    def __init__(
        self,
        source: str | Path,
        start_sec: float = 0.0,
        frame_stride: int = 1,
    ) -> None:
        self._source = str(source)
        self._start_sec = start_sec
        self._frame_stride = max(1, frame_stride)

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self._source)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 12.0)
            raw_index = max(0, int(max(self._start_sec, 0.0) * max(fps, 1.0)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, raw_index)
            frame_index = 0
            while True:
                read_ok, frame_bgr = capture.read()
                if not read_ok:
                    break
                if raw_index % self._frame_stride == 0:
                    time_sec = raw_index / max(fps, 1.0)
                    image = np.asarray(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                        dtype=np.uint8,
                    )
                    yield Frame(index=frame_index, time_sec=round(time_sec, 3), image=image)
                    frame_index += 1
                raw_index += 1
        finally:
            capture.release()


__all__ = ["VideoFileSource"]
