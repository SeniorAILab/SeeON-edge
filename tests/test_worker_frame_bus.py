"""RED contract for the concrete per-camera bounded frame bus.

Assumed public API:

* ``BoundedFrameBus(evidence_capacity=128, clock=...)`` constructs one bus for
  one camera.
* ``inference``, ``live``, and ``evidence`` expose canonical subscriptions.
  Their capacities are 1, 1, and ``evidence_capacity`` respectively; the
  first two use latest-only replacement and evidence uses FIFO drop-newest.
* ``subscribe(name, capacity=..., latest_only=...)`` remains available for
  named subscriptions.
* ``metrics(name)`` returns an immutable snapshot with ``published``, ``taken``,
  ``dropped``, and ``queue_age_sec`` fields.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

from contracts.frame import Frame
from worker.interfaces.bus import FrameBus
from worker.pipeline.bus import BoundedFrameBus
from worker.types import FramePacket


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now: float = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_packet(seq: int, *, camera_id: str = "camera-a") -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(index=seq, time_sec=float(seq), image=image),
        pts=float(seq),
        seq=seq,
        width=3,
        height=2,
        decode_time_ms=1.5,
    )


def test_canonical_subscriptions_have_required_capacities_and_policies() -> None:
    # Given a bus with the default configured evidence capacity
    bus = BoundedFrameBus()
    assert isinstance(bus, FrameBus)

    # Then canonical subscriptions expose inference/latest-only/cap1,
    # live/latest-only/cap1, and evidence/FIFO/cap128.
    assert bus.inference.capacity == 1
    assert bus.inference.latest_only is True
    assert bus.live.capacity == 1
    assert bus.live.latest_only is True
    assert bus.evidence.capacity == 128
    assert bus.evidence.latest_only is False


def test_canonical_evidence_capacity_is_configurable() -> None:
    # Given a bus configured with an explicit evidence capacity
    bus = BoundedFrameBus(evidence_capacity=3)

    # Then only the evidence subscription adopts that configured capacity.
    assert bus.evidence.capacity == 3
    assert bus.inference.capacity == 1
    assert bus.live.capacity == 1


def test_latest_only_replaces_oldest_pending_packet_and_counts_one_drop() -> None:
    # Given an empty latest-only subscription
    bus = BoundedFrameBus()
    first = make_packet(1)
    second = make_packet(2)

    # When two packets are published before the consumer takes one
    bus.publish(first)
    bus.publish(second)

    # Then only the newest packet remains and exactly one packet was dropped.
    assert bus.inference.take(timeout_sec=0) is second
    assert bus.inference.take(timeout_sec=0) is None
    assert bus.metrics("inference").dropped == 1


def test_evidence_full_queue_drops_newest_and_preserves_fifo_order() -> None:
    # Given a FIFO evidence subscription with capacity two
    bus = BoundedFrameBus(evidence_capacity=2)
    first, second, newest = make_packet(1), make_packet(2), make_packet(3)

    # When a third packet arrives while the queue is full
    bus.publish(first)
    bus.publish(second)
    bus.publish(newest)

    # Then the newest packet is dropped and the existing packets remain FIFO.
    assert bus.evidence.take(timeout_sec=0) is first
    assert bus.evidence.take(timeout_sec=0) is second
    assert bus.evidence.take(timeout_sec=0) is None
    assert bus.metrics("evidence").dropped == 1


def test_metrics_report_exact_publish_take_and_drop_counts() -> None:
    # Given packets that exercise replacement, successful take, and empty take
    bus = BoundedFrameBus(evidence_capacity=1)
    bus.publish(make_packet(1))
    bus.publish(make_packet(2))
    assert bus.inference.take(timeout_sec=0) is not None
    assert bus.inference.take(timeout_sec=0) is None

    # Then metrics distinguish publishes, takes, and drops exactly.
    snapshot = bus.metrics("inference")
    assert (snapshot.published, snapshot.taken, snapshot.dropped) == (2, 1, 1)


def test_queue_age_uses_injected_clock_deterministically() -> None:
    # Given a fake clock and one queued packet
    clock = FakeClock(100.0)
    bus = BoundedFrameBus(clock=clock)
    bus.publish(make_packet(1))

    # When time advances without consuming the packet
    clock.advance(2.5)

    # Then the metric reports the deterministic age of the oldest queued item.
    assert bus.metrics("inference").queue_age_sec == 2.5

    # And consuming it removes the queue age.
    assert bus.inference.take(timeout_sec=0) is not None
    assert bus.metrics("inference").queue_age_sec == 0.0


def test_subscriber_counters_are_independent() -> None:
    # Given two named subscriptions on one bus
    bus = BoundedFrameBus()
    inference = bus.subscribe("custom-inference", capacity=1, latest_only=True)
    _ = bus.subscribe("custom-evidence", capacity=2)

    # When both receive the same publication but only one is consumed
    bus.publish(make_packet(1))
    assert inference.take(timeout_sec=0) is not None

    # Then each subscription reports only its own counters.
    assert (
        bus.metrics("custom-inference").published,
        bus.metrics("custom-inference").taken,
    ) == (1, 1)
    assert (
        bus.metrics("custom-evidence").published,
        bus.metrics("custom-evidence").taken,
    ) == (1, 0)


def test_independent_per_camera_buses_do_not_affect_each_other() -> None:
    # Given separate concrete buses representing cameras A and B
    camera_a = BoundedFrameBus(evidence_capacity=1)
    camera_b = BoundedFrameBus(evidence_capacity=1)

    # When camera A's slow evidence consumer causes a drop
    camera_a.publish(make_packet(1, camera_id="a"))
    camera_a.publish(make_packet(2, camera_id="a"))
    camera_b.publish(make_packet(10, camera_id="b"))

    # Then camera B remains able to receive its own packet unaffected.
    assert camera_b.evidence.take(timeout_sec=0) is not None
    assert camera_a.metrics("evidence").dropped == 1
    assert camera_b.metrics("evidence").dropped == 0


def test_all_subscribers_receive_the_same_immutable_packet_identity() -> None:
    # Given canonical subscribers and one immutable packet
    bus = BoundedFrameBus()
    packet = make_packet(7)

    # When the packet is published once
    bus.publish(packet)

    # Then every subscriber observes the exact same packet object.
    assert bus.inference.take(timeout_sec=0) is packet
    assert bus.live.take(timeout_sec=0) is packet
    assert bus.evidence.take(timeout_sec=0) is packet


def test_mutating_a_consumer_image_copy_leaves_source_and_shared_packet_unchanged() -> None:
    # Given one packet shared with two consumers
    bus = BoundedFrameBus()
    packet = make_packet(4)
    original = packet.frame.image.copy()
    bus.publish(packet)

    # When a consumer copies the image before mutating its working buffer
    inference_packet = bus.inference.take(timeout_sec=0)
    assert inference_packet is not None
    assert inference_packet is packet
    consumer_image = inference_packet.frame.image.copy()
    consumer_image[0, 0, 0] = 255

    # Then the source frame and the other subscriber's packet remain unchanged.
    assert np.array_equal(packet.frame.image, original)
    live_packet = bus.live.take(timeout_sec=0)
    assert live_packet is not None
    assert live_packet is packet
    assert np.array_equal(live_packet.frame.image, original)


def test_empty_nonblocking_take_returns_none() -> None:
    # Given an empty subscription
    bus = BoundedFrameBus()

    # When the consumer requests a nonblocking take
    result = bus.inference.take(timeout_sec=0)

    # Then no packet is fabricated and the call returns immediately.
    assert result is None


def test_metrics_snapshot_is_immutable() -> None:
    # Given a metrics snapshot
    bus = BoundedFrameBus()
    snapshot = bus.metrics("inference")

    # Then the snapshot is a frozen dataclass rather than mutable live state.
    assert is_dataclass(snapshot)
    field_name = "published"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        setattr(snapshot, field_name, 99)
