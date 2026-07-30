from __future__ import annotations

import threading

import numpy as np

from contracts.frame import Frame
from edge.runtime.latest_frame import LatestFrameBuffer


def test_latest_frame_buffer_overwrites_stale_pending_frame() -> None:
    buffer = LatestFrameBuffer()
    stop_event = threading.Event()
    finished = threading.Event()
    result: list[bool] = []

    assert buffer.put(_frame(1), stop_event=stop_event)

    def put_newer_frame() -> None:
        result.append(buffer.put(_frame(3), stop_event=stop_event))
        finished.set()

    thread = threading.Thread(target=put_newer_frame)
    thread.start()
    try:
        assert finished.wait(timeout=0.1)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    latest = buffer.take(timeout_sec=0.01)
    assert result == [True]
    assert latest is not None
    assert latest.index == 3


def _frame(index: int) -> Frame:
    image = np.full((1, 1, 3), index, dtype=np.uint8)
    return Frame(index=index, time_sec=float(index), image=image)
