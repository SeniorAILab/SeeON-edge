from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from contracts.frame import Frame


class CameraSource:
    """Frame source over a live camera device (frame_source.cv2.VideoCapture by index).

    Implements the FrameSource protocol for real-time capture so the downstream
    iter_live_frames loop requires no changes — camera and file sources share
    the same contract.

    Unlike VideoFileSource there is no fps metadata to trust for time_sec;
    time_sec uses time.monotonic() elapsed since the first frame so timestamps
    remain meaningful even when the camera feed stutters.

    CAP_PROP_BUFFERSIZE=1 keeps the internal queue at one frame so each read
    returns the most recently captured image rather than a stale buffered one.
    """

    def __init__(self, device_index: int, max_failures: int = 30) -> None:
        self._device_index = device_index
        self._max_failures = max_failures

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self._device_index)
        # Prefer the latest frame: drop stale buffer so each read returns fresh data.
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
                time_sec = round(now - t0, 3)
                image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield Frame(index=frame_index, time_sec=time_sec, image=image)
                frame_index += 1
        finally:
            capture.release()


__all__ = ["CameraSource"]
