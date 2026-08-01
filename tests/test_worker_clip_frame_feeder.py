"""Unit coverage for ``ClipFrameFeeder``: ``bus.evidence`` -> ``ClipRecorder``.

Covers the properties the composition root relies on but does not itself
prove: each drained packet's ``Frame`` payload actually reaches
``recorder.on_frame(camera_id, frame)`` and ``fed_count`` tracks it, a poll
timeout (``take()`` returning ``None``) is neither counted as a feed nor
treated as a reason to exit, a raising ``recorder.on_frame`` is isolated to
that one packet without killing the loop, and ``stop()`` actually terminates
a running ``run()``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import final

import numpy as np

from contracts.frame import Frame
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.output.evidence.clip_frame_feeder import ClipFrameFeeder
from worker.types import FramePacket


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


def _wait_for(predicate: Callable[[], bool], *, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@final
class _RecordingRecorder:
    """Fake ``ClipRecorder``: records every ``on_frame`` call it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Frame]] = []

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        self.calls.append((camera_id, frame))
        return True


@final
class _RaisingOnFirstCallRecorder:
    """Fake ``ClipRecorder`` whose first ``on_frame`` call raises, exercising
    the feeder's per-packet failure isolation; every later call succeeds and
    is recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Frame]] = []
        self._raised = False

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        if not self._raised:
            self._raised = True
            raise RuntimeError("recorder admission exploded")
        self.calls.append((camera_id, frame))
        return True


def test_run_forwards_each_packets_frame_to_recorder_on_frame_and_increments_fed_count() -> None:
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    packet = _packet("cam-a", 1)
    bus.publish(packet)
    assert _wait_for(lambda: feeder.fed_count >= 1)
    feeder.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert feeder.fed_count == 1
    assert recorder.calls == [("cam-a", packet.frame)]


def test_poll_timeout_returning_none_does_not_count_as_a_feed_and_the_loop_keeps_polling() -> None:
    """``take()`` returns ``None`` on every poll while the bus stays empty;
    ``run()`` must not treat that as a fed frame or exit on it -- it keeps
    polling until ``stop()`` is called."""
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    # Several poll-timeout cycles with nothing published: proves the loop
    # survives repeated `None` returns instead of exiting after the first one.
    time.sleep(0.1)
    assert thread.is_alive()
    assert feeder.fed_count == 0
    assert recorder.calls == []

    feeder.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_a_raising_recorder_on_frame_is_isolated_per_packet_and_the_loop_continues() -> None:
    bus = BoundedFrameBus()
    recorder = _RaisingOnFirstCallRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    first = _packet("cam-a", 1)
    bus.publish(first)
    assert _wait_for(lambda: feeder.failure_count >= 1)

    second = _packet("cam-a", 2)
    bus.publish(second)
    assert _wait_for(lambda: feeder.fed_count >= 1)

    feeder.stop()
    thread.join(timeout=2.0)

    # If the exception from the first packet had escaped `run()`, the thread
    # would have died right there instead of reaching `stop()`'s join.
    assert not thread.is_alive()
    assert feeder.failure_count == 1
    assert feeder.fed_count == 1
    assert recorder.calls == [("cam-a", second.frame)]


def test_stop_terminates_the_loop_and_run_returns() -> None:
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()
    assert thread.is_alive()

    feeder.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
