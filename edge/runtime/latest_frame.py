from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from contracts.frame import Frame


@dataclass(slots=True)
class LatestFrameBuffer:
    _queue: queue.Queue[Frame] = field(default_factory=lambda: queue.Queue(maxsize=1))

    def put(self, frame: Frame, *, stop_event: threading.Event) -> bool:
        while not stop_event.is_set():
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
                self._queue.task_done()
            else:
                return True
        return False

    def take(self, *, timeout_sec: float) -> Frame | None:
        try:
            frame = self._queue.get(timeout=timeout_sec)
        except queue.Empty:
            return None
        self._queue.task_done()
        return frame


__all__ = ["LatestFrameBuffer"]
