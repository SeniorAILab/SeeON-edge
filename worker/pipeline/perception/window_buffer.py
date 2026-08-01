from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TypeAlias, final

PoseVector: TypeAlias = Sequence[float]
Window: TypeAlias = tuple[tuple[float, ...], ...]


@final
class WindowBuffer:
    """Collect numeric pose vectors and emit fixed windows at a stable stride."""

    def __init__(self, window: int, stride: int) -> None:
        if window < 1:
            message = "window must be >= 1"
            raise ValueError(message)
        if stride < 1:
            message = "stride must be >= 1"
            raise ValueError(message)
        self.window: int = window
        self.stride: int = stride
        self._poses: deque[tuple[float, ...]] = deque(maxlen=window)
        self._appended: int = 0
        self._last_emit_at: int = 0

    def append(self, pose: PoseVector) -> tuple[Window, ...]:
        """Append one vector and return the newly due window, if any."""
        self._poses.append(tuple(float(value) for value in pose))
        self._appended += 1
        if len(self._poses) < self.window:
            return ()
        if self._appended - self._last_emit_at < self.stride:
            return ()
        self._last_emit_at = self._appended
        return (tuple(self._poses),)

    def clear(self) -> None:
        self._poses.clear()
        self._appended = 0
        self._last_emit_at = 0


__all__ = ["WindowBuffer"]
