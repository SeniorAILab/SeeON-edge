"""L4 event emission package."""

from __future__ import annotations

__all__ = [
    "EventPublisher",
    "LoggingEventPublisher",
    "Outbox",
    "StubEventPublisher",
]


def __getattr__(name: str) -> object:
    if name in {"EventPublisher", "LoggingEventPublisher", "StubEventPublisher"}:
        from shared.events import local_publisher

        return getattr(local_publisher, name)
    if name == "Outbox":
        from shared.events.outbox import Outbox

        return Outbox
    raise AttributeError(name)
