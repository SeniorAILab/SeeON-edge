from __future__ import annotations

import logging
from typing import Protocol

from shared.events.schemas import EmittedEvent

_LOGGER = logging.getLogger(__name__)


class EventPublisher(Protocol):
    def publish(self, event: EmittedEvent) -> None: ...


class LoggingEventPublisher:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = _LOGGER if logger is None else logger
        self.published: list[EmittedEvent] = []

    def publish(self, event: EmittedEvent) -> None:
        self.published.append(event)
        self._logger.info("ml.event published", extra={"event": event.as_dict()})


StubEventPublisher = LoggingEventPublisher


__all__ = [
    "EventPublisher",
    "LoggingEventPublisher",
    "StubEventPublisher",
]
