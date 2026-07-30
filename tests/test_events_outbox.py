from __future__ import annotations

from shared.events import LoggingEventPublisher, Outbox
from shared.events.schemas import build_emitted_event


def _event(event_type: str = "fall"):
    return build_emitted_event(
        facility="facility-001",
        camera="cam-001",
        domain="fall_detection",
        event_type=event_type,
        severity="HIGH",
        evidence={"confidence": 0.9},
    )


def test_outbox_buffers_and_flushes_events_to_stub_publisher() -> None:
    publisher = LoggingEventPublisher()
    outbox = Outbox(publisher=publisher)
    event = _event()

    assert outbox.buffer(event) is True
    assert outbox.pending_count == 1

    assert outbox.flush() == 1
    assert publisher.published == [event]
    assert outbox.pending_count == 0
    assert outbox.failure_count == 0


def test_outbox_flush_is_idempotent_after_queue_is_empty() -> None:
    publisher = LoggingEventPublisher()
    outbox = Outbox(publisher=publisher)
    event = _event("bed-exit")

    outbox.buffer(event)
    assert outbox.flush() == 1
    assert outbox.flush() == 0

    assert publisher.published == [event]
    assert outbox.pending_count == 0


def test_outbox_counts_full_queue_drops() -> None:
    publisher = LoggingEventPublisher()
    outbox = Outbox(publisher=publisher, max_size=1)

    assert outbox.buffer(_event("fall")) is True
    assert outbox.buffer(_event("bed-exit")) is False

    assert outbox.pending_count == 1
    assert outbox.drop_count == 1
    assert outbox.failure_count == 1
