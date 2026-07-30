"""Sliding pose-window buffer for temporal perception paths."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

PoseVector = Sequence[float]
Window = tuple[tuple[float, ...], ...]


class WindowBuffer:
    """Collect pose vectors and emit fixed-length windows at a configured stride."""

    def __init__(self, window: int, stride: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.window = window
        self.stride = stride
        self._poses: deque[tuple[float, ...]] = deque(maxlen=window)
        self._appended = 0
        self._last_emit_at = 0

    def append(self, pose: PoseVector) -> tuple[Window, ...]:
        """Append one pose vector and return zero or one newly due windows."""
        self._poses.append(tuple(float(value) for value in pose))
        self._appended += 1
        if len(self._poses) < self.window:
            return ()
        if self._appended - self._last_emit_at < self.stride:
            return ()
        self._last_emit_at = self._appended
        return (tuple(self._poses),)

    def clear(self) -> None:
        """Drop buffered poses and reset stride accounting."""
        self._poses.clear()
        self._appended = 0
        self._last_emit_at = 0
