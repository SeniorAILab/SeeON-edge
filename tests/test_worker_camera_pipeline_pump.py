"""Per-camera pipeline pump: ``bus.inference`` -> analytics -> decision -> sink.

Covers the properties the composition root relies on but does not itself
prove: admitted events actually reach the sink through a real ``emit()``
call, the scheduler still gates which extractor runs, one camera's pump
failure never stops another camera's pump, and ``IngestSupervisor.stop()``
actually reaps real pump threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.model.errors import FatalAcceleratorError
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.ingest.lifecycle import IngestSupervisor
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.types import BusinessEvent, DecisionInput, FramePacket, ModuleResult


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


def _blank_analytics(
    camera_id: str,
    *,
    extractors: tuple[object, ...] = (),
    task_intervals: dict[str, int] | None = None,
) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=extractors,  # type: ignore[arg-type]
        scheduler=Scheduler(task_intervals=task_intervals or {}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id=camera_id),
    )


def _wait_for(predicate: Callable[[], bool], *, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@final
class _RecordingExtractor:
    """Fake ``NamedExtractor``: records exactly which frames it was asked to extract."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.calls: list[int] = []

    def extract(self, packet: FramePacket) -> ModuleResult:
        self.calls.append(packet.frame.index)
        return ModuleResult(module_name=self.module_name, result=(), elapsed_ms=0.0)


@final
class _FrameEchoDecider:
    """Emits one uniquely-identified event per decision input it sees."""

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return (
            BusinessEvent(
                domain="probe",
                event_type="tick",
                identity=f"frame-{input_value.frame_index}",
                camera_id=self._camera_id,
                facility_id=f"facility-{self._camera_id}",
                time_sec=input_value.time_sec or 0.0,
                probability=1.0,
            ),
        )


@final
class _RaisingDecider:
    """Always fails: exercises the pump's per-frame failure isolation branch."""

    def __init__(self) -> None:
        self.call_count = 0

    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.call_count += 1
        raise RuntimeError("decision stage exploded")


@final
class _FatalDecider:
    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        raise FatalAcceleratorError(
            "injected accelerator failure", camera_id="camera-a", task="pose"
        )


@final
class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[BusinessEvent] = []
        self.on_emit: Callable[[BusinessEvent], None] | None = None

    def emit(self, event: BusinessEvent) -> None:
        self.events.append(event)
        if self.on_emit is not None:
            self.on_emit(event)


def test_pump_forwards_an_admitted_event_from_the_bus_to_the_sink_via_real_emit() -> None:
    """End-to-end: bus.publish -> take -> analytics.process -> decision.update -> sink.emit."""
    bus = BoundedFrameBus()
    extractor = _RecordingExtractor("pose")
    analytics = _blank_analytics(
        "camera-a", extractors=(extractor,), task_intervals={"pose": 1}
    )
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a", bus.inference, analytics, decision, sink, poll_timeout_sec=0.02
    )
    sink.on_emit = lambda _event: pump.stop()

    bus.publish(_packet("camera-a", 1))
    pump.run()

    assert extractor.calls == [1]
    assert len(sink.events) == 1
    admitted = sink.events[0]
    assert admitted.domain == "probe"
    assert admitted.event_type == "tick"
    assert admitted.camera_id == "camera-a"
    # IncidentManager.admit() resolves the decider's identity into a durable
    # edge_event_id, so only its presence (not its exact value) is asserted here.
    assert isinstance(admitted.identity, str) and admitted.identity


def test_scheduler_gates_extraction_so_skipped_frames_never_reach_the_extractor() -> None:
    """analytics.process only calls extractors for modules the scheduler marks due."""
    bus = BoundedFrameBus()
    extractor = _RecordingExtractor("probe")
    # interval 2: frame 1 is skipped (1 % 2 != 0), frame 2 is due (2 % 2 == 0).
    analytics = _blank_analytics(
        "camera-a", extractors=(extractor,), task_intervals={"probe": 2}
    )
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a", bus.inference, analytics, decision, sink, poll_timeout_sec=0.02
    )

    def _on_emit(_event: BusinessEvent) -> None:
        if len(sink.events) == 1:
            bus.publish(_packet("camera-a", 2))
        else:
            pump.stop()

    sink.on_emit = _on_emit
    bus.publish(_packet("camera-a", 1))
    pump.run()

    # Decision still runs on every frame (identity/track state must advance regardless),
    # but the scheduled-out module is never asked to extract the skipped frame.
    assert len(sink.events) == 2
    assert extractor.calls == [2]


def test_one_camera_pump_failure_does_not_stop_the_other_camera_pump() -> None:
    bus_a = BoundedFrameBus()
    bus_b = BoundedFrameBus()
    raising = _RaisingDecider()
    sink_a = _RecordingSink()
    decision_a = EventAggregator(deciders=(raising,), incidents=IncidentManager())
    pump_a = CameraPipelinePump(
        "camera-a", bus_a.inference, _blank_analytics("camera-a"), decision_a, sink_a,
        poll_timeout_sec=0.02,
    )

    sink_b = _RecordingSink()
    decision_b = EventAggregator(
        deciders=(_FrameEchoDecider("camera-b"),), incidents=IncidentManager()
    )
    pump_b = CameraPipelinePump(
        "camera-b", bus_b.inference, _blank_analytics("camera-b"), decision_b, sink_b,
        poll_timeout_sec=0.02,
    )

    supervisor = IngestSupervisor([pump_a, pump_b])
    supervisor.start()
    try:
        bus_a.publish(_packet("camera-a", 1))
        bus_b.publish(_packet("camera-b", 1))

        assert _wait_for(lambda: len(sink_b.events) >= 1)
        assert _wait_for(lambda: pump_a.failure_count >= 1)

        # Publish a second frame to camera-a: if the exception had killed the
        # thread instead of being isolated, this would never be processed.
        bus_a.publish(_packet("camera-a", 2))
        assert _wait_for(lambda: pump_a.failure_count >= 2)
    finally:
        supervisor.stop(join_timeout_sec=2.0)

    assert pump_a.failure_count == 2
    assert raising.call_count == 2
    assert len(sink_b.events) == 1
    assert sink_a.events == []


def test_fatal_accelerator_error_propagates_out_of_run_instead_of_being_swallowed() -> None:
    bus = BoundedFrameBus()
    decision = EventAggregator(deciders=(_FatalDecider(),), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a", bus.inference, _blank_analytics("camera-a"), decision, sink,
        poll_timeout_sec=0.02,
    )

    bus.publish(_packet("camera-a", 1))

    with pytest.raises(FatalAcceleratorError):
        pump.run()


def test_supervisor_stop_joins_real_pump_threads() -> None:
    bus = BoundedFrameBus()
    decision = EventAggregator(deciders=(), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a", bus.inference, _blank_analytics("camera-a"), decision, sink,
        poll_timeout_sec=0.02,
    )
    supervisor = IngestSupervisor([pump])

    supervisor.start()
    thread_name = "worker-ingest-camera-a"
    started_thread = next(
        thread for thread in threading.enumerate() if thread.name == thread_name
    )
    assert started_thread.is_alive()

    supervisor.stop(join_timeout_sec=2.0)

    assert not started_thread.is_alive()
