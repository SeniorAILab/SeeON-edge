"""Internal per-name bounded subscription backing BoundedFrameBus.

Not part of the public frame-bus surface; consumers only see the
``FrameSubscription`` shape (``capacity``, ``latest_only``, ``take``,
``close``) exposed through ``BoundedFrameBus``.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from worker.pipeline.bus.metrics import BusMetricsSnapshot
from worker.types import FramePacket


class BoundedSubscription:
    """One capacity-bounded, thread-safe fan-out queue for a single subscriber.

    ``latest_only`` subscriptions evict the oldest queued packet to make room
    for the newest one (drop-oldest). FIFO subscriptions (``latest_only=False``,
    e.g. evidence) drop the incoming packet instead once full (drop-newest),
    preserving already-queued order.
    """

    __slots__ = (
        "capacity",
        "latest_only",
        "_clock",
        "_condition",
        "_queue",
        "_published",
        "_taken",
        "_dropped",
        "_closed",
    )

    def __init__(
        self,
        *,
        capacity: int,
        latest_only: bool,
        clock: Callable[[], float],
    ) -> None:
        self.capacity = capacity
        self.latest_only = latest_only
        self._clock = clock
        self._condition = threading.Condition()
        self._queue: deque[tuple[FramePacket, float]] = deque()
        self._published = 0
        self._taken = 0
        self._dropped = 0
        self._closed = False

    def publish(self, packet: FramePacket) -> None:
        with self._condition:
            self._published += 1
            if len(self._queue) >= self.capacity:
                self._dropped += 1
                if self.latest_only:
                    self._queue.popleft()
                    self._queue.append((packet, self._clock()))
                # FIFO (evidence-style) subscriptions drop the incoming
                # packet and keep the already-queued order untouched.
            else:
                self._queue.append((packet, self._clock()))
            self._condition.notify_all()

    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        with self._condition:
            if not self._queue and not self._closed and timeout_sec != 0:
                self._condition.wait_for(
                    lambda: bool(self._queue) or self._closed,
                    timeout=timeout_sec,
                )
            if not self._queue:
                return None
            packet, _published_at = self._queue.popleft()
            self._taken += 1
            return packet

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def metrics(self) -> BusMetricsSnapshot:
        with self._condition:
            queue_age_sec = 0.0
            if self._queue:
                _oldest_packet, oldest_published_at = self._queue[0]
                queue_age_sec = self._clock() - oldest_published_at
            return BusMetricsSnapshot(
                published=self._published,
                taken=self._taken,
                dropped=self._dropped,
                queue_age_sec=queue_age_sec,
            )


__all__ = ["BoundedSubscription"]
