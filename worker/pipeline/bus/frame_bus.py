"""Concrete per-camera bounded frame bus.

Fans out immutable ``FramePacket`` instances to independently bounded,
named subscriptions. Canonical subscriptions match the legacy topology:

* ``inference`` -- latest-only, capacity 1
* ``live`` -- latest-only, capacity 1
* ``evidence`` -- FIFO drop-newest, capacity ``evidence_capacity`` (default 128)

Additional named subscriptions can be created via ``subscribe`` for tests
or specialized consumers; each gets its own independent counters.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic

from worker.pipeline.bus.metrics import BusMetricsSnapshot
from worker.pipeline.bus.subscription import BoundedSubscription
from worker.types import FramePacket

_DEFAULT_EVIDENCE_CAPACITY = 128


class BoundedFrameBus:
    """Per-camera frame bus: one immutable packet fanned out to N subscribers."""

    __slots__ = ("_clock", "_lock", "_subscriptions", "inference", "live", "evidence")

    def __init__(
        self,
        *,
        evidence_capacity: int = _DEFAULT_EVIDENCE_CAPACITY,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._subscriptions: dict[str, BoundedSubscription] = {}
        self.inference = self.subscribe("inference", capacity=1, latest_only=True)
        self.live = self.subscribe("live", capacity=1, latest_only=True)
        self.evidence = self.subscribe(
            "evidence", capacity=evidence_capacity, latest_only=False
        )

    def subscribe(
        self,
        name: str,
        *,
        capacity: int,
        latest_only: bool = False,
    ) -> BoundedSubscription:
        with self._lock:
            subscription = BoundedSubscription(
                capacity=capacity, latest_only=latest_only, clock=self._clock
            )
            self._subscriptions[name] = subscription
            return subscription

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            subscriptions = tuple(self._subscriptions.values())
        for subscription in subscriptions:
            subscription.publish(packet)

    def metrics(self, name: str) -> BusMetricsSnapshot:
        with self._lock:
            subscription = self._subscriptions[name]
        return subscription.metrics()

    def close(self) -> None:
        with self._lock:
            subscriptions = tuple(self._subscriptions.values())
        for subscription in subscriptions:
            subscription.close()


__all__ = ["BoundedFrameBus"]
