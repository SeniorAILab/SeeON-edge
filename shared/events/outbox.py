"""Buffered event outbox with explicit flush semantics.

ML emits typed signed events. Backend owns severity/channel/policy/final-dedup;
ML runtime incident management owns only idempotency and cooldown.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field

from shared.events.local_publisher import EventPublisher
from shared.events.schemas import EmittedEvent


@dataclass(slots=True)
class Outbox:
    publisher: EventPublisher
    max_size: int = 128
    _queue: queue.Queue[EmittedEvent] = field(init=False, repr=False)
    failure_count: int = 0
    drop_count: int = 0

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=max(1, self.max_size))

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def buffer(self, event: EmittedEvent) -> bool:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.drop_count += 1
            self.failure_count += 1
            return False
        return True

    def flush(self) -> int:
        published = 0
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return published
            try:
                self.publisher.publish(event)
                published += 1
            except Exception:
                self.failure_count += 1
                self._queue.put_nowait(event)
                raise
            finally:
                self._queue.task_done()


__all__ = ["Outbox"]
