from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np

from contracts.frame import Frame
from contracts.observation import FrameObservation
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.scheduler import Scheduler


class _Detector:
    def update(self, observation: FrameObservation, time_sec: float | None = None):
        del observation, time_sec
        return {"event_type": "bed-exit", "probability": 0.9, "event_id": "evt-1"}

class _SequenceDetector:
    def update(self, observation: FrameObservation, time_sec: float | None = None):
        del observation, time_sec
        return [
            {"event_type": "bed-exit", "probability": 0.9, "event_id": "evt-1"},
            {"event_type": "bed-exit", "probability": 0.9, "event_id": "evt-2"},
        ]
class _IdentitylessSequenceDetector:
    def update(self, observation: FrameObservation, time_sec: float | None = None):
        del observation, time_sec
        return [
            {"event_type": "bed-exit", "probability": 0.9},
            {"event_type": "bed-exit", "probability": 0.9},
        ]

class _TwoFrameDetector:
    def __init__(self) -> None:
        self._events = iter(
            (
                {"event_type": "bed-exit", "probability": 0.9, "event_id": "evt-1"},
                {"event_type": "bed-exit", "probability": 0.9, "event_id": "evt-2"},
            )
        )

    def update(self, observation: FrameObservation, time_sec: float | None = None):
        del observation, time_sec
        return next(self._events)




@dataclass(slots=True)
class _Sink:
    identity_path: Path | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    persisted_before_emit: bool = False

    def emit(self, event: dict[str, object]) -> None:
        if self.identity_path is not None:
            self.persisted_before_emit = (
                str(event["edge_event_id"])
                in self.identity_path.read_text(encoding="utf-8")
            )
        self.events.append(dict(event))


@dataclass(slots=True)
class _Recorder:
    clip_id: str | None
    identity_path: Path | None = None
    event_refs: list[tuple[str, str]] = field(default_factory=list)
    event_types: list[str | None] = field(default_factory=list)
    allow_new_clips: list[bool] = field(default_factory=list)
    persisted_before_event: bool = False

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        del camera_id, frame
        return True

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        if self.identity_path is not None:
            self.persisted_before_event = (
                event_ref in self.identity_path.read_text(encoding="utf-8")
            )
        self.event_refs.append((camera_id, event_ref))
        self.event_types.append(event_type)
        self.allow_new_clips.append(allow_new_clip)
        return self.clip_id

@dataclass(slots=True)
class _AttachOnlyRecorder:
    active_clip_id: str | None = None

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        del camera_id, frame
        return True

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        del camera_id, event_ref, event_type
        if self.active_clip_id is not None:
            return self.active_clip_id
        if not allow_new_clip:
            return None
        self.active_clip_id = "clip-123"
        return self.active_clip_id


@dataclass(slots=True)
class _Stager:
    calls: list[str] = field(default_factory=list)

    def stage(self, event: dict[str, object]) -> None:
        self.calls.append(f"stage:{event['edge_event_id']}")

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.calls.append(f"complete:{edge_event_id}:{clip_id}")


@dataclass(slots=True)
class _OrderingRecorder(_Recorder):
    stager: _Stager | None = None

    def on_event(self, *args, **kwargs) -> str | None:
        assert self.stager is not None and self.stager.calls[0].startswith("stage:")
        self.stager.calls.append("recorder")
        return _Recorder.on_event(self, *args, **kwargs)



def _frame() -> Frame:
    return Frame(index=0, time_sec=12.0, image=np.zeros((8, 8, 3), dtype=np.uint8))


def test_camera_worker_persists_canonical_event_identity_before_side_effects(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "event-identities.jsonl"
    sink = _Sink(identity_path)
    recorder = _Recorder("clip-123", identity_path)
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=sink,
        clip_recorder=recorder,
        event_identity_path=identity_path,
    )

    worker.process_frame(_frame())

    edge_event_id = str(sink.events[0]["edge_event_id"])
    parsed_id = UUID(edge_event_id)
    assert parsed_id.version == 4
    assert str(parsed_id) == edge_event_id
    assert str(sink.events[0]["detected_at"]).endswith("Z")
    assert recorder.event_refs == [("cam-1", edge_event_id)]
    assert edge_event_id in identity_path.read_text(encoding="utf-8")
    assert recorder.persisted_before_event
    assert sink.persisted_before_emit


def test_camera_worker_reuses_persisted_identity_after_restart(tmp_path: Path) -> None:
    identity_path = tmp_path / "event-identities.jsonl"
    first_sink = _Sink()
    first_worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=first_sink,
        event_identity_path=identity_path,
    )
    first_worker.process_frame(_frame())

    restarted_sink = _Sink()
    restarted_worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=restarted_sink,
        event_identity_path=identity_path,
    )
    restarted_worker.process_frame(_frame())

    assert restarted_sink.events[0]["edge_event_id"] == first_sink.events[0]["edge_event_id"]
    assert restarted_sink.events[0]["detected_at"] == first_sink.events[0]["detected_at"]
    assert len(identity_path.read_text(encoding="utf-8").splitlines()) == 1


def test_camera_worker_propagates_recorder_clip_id_on_event() -> None:
    sink = _Sink()
    recorder = _Recorder("clip-123")
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=sink,
        clip_recorder=recorder,
    )

    worker.process_frame(_frame())

    assert recorder.event_refs == [
        ("cam-1", str(sink.events[0]["edge_event_id"]))
    ]
    assert recorder.event_types == ["bed-exit"]
    assert sink.events[0]["clip_id"] == "clip-123"


def test_camera_worker_stages_before_recorder_and_skips_immediate_network() -> None:
    sink = _Sink()
    stager = _Stager()
    recorder = _OrderingRecorder("clip-123", stager=stager)
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=sink,
        clip_recorder=recorder,
        evidence_stager=stager,
    )

    worker.process_frame(_frame())

    assert sink.events == []
    assert len(stager.calls) == 3
    assert stager.calls[0].startswith("stage:")
    assert stager.calls[1] == "recorder"
    event_id = stager.calls[0].removeprefix("stage:")
    assert stager.calls[2] == f"complete:{event_id}:clip-123"


def test_camera_worker_sets_null_clip_id_without_recorder() -> None:
    sink = _Sink()
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=sink,
    )

    worker.process_frame(_frame())

    assert sink.events[0]["clip_id"] is None
def test_camera_worker_emits_distinct_events_while_throttling_clip_recording(monkeypatch) -> None:
    sink = _Sink()
    recorder = _Recorder("clip-123")
    times = iter((100.0, 100.0, 101.0, 101.0))
    monkeypatch.setattr("edge.runtime.camera_worker.time.monotonic", lambda: next(times))
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_SequenceDetector(),),
        event_sink=sink,
        clip_recorder=recorder,
        clip_recording_min_interval_sec=30.0,
    )

    worker.process_frame(_frame())

    assert [event["event_id"] for event in sink.events] == ["evt-1", "evt-2"]
    assert recorder.event_refs == [
        ("cam-1", str(event["edge_event_id"]))
        for event in sink.events
    ]
    assert recorder.allow_new_clips == [True, False]
    assert [event["clip_id"] for event in sink.events] == ["clip-123", "clip-123"]
def test_camera_worker_throttle_does_not_open_clip_after_active_clip_ends(monkeypatch) -> None:
    sink = _Sink()
    recorder = _AttachOnlyRecorder()
    times = iter((100.0, 100.0, 101.0, 101.0))
    monkeypatch.setattr("edge.runtime.camera_worker.time.monotonic", lambda: next(times))
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_TwoFrameDetector(),),
        event_sink=sink,
        clip_recorder=recorder,
        clip_recording_min_interval_sec=30.0,
    )

    worker.process_frame(_frame())
    recorder.active_clip_id = None
    worker.process_frame(_frame())

    assert [event["clip_id"] for event in sink.events] == ["clip-123", None]

def test_camera_worker_emits_identityless_events_within_incident_cooldown(monkeypatch) -> None:
    sink = _Sink()
    times = iter((100.0, 101.0))
    monkeypatch.setattr("edge.runtime.camera_worker.time.monotonic", lambda: next(times))
    worker = CameraWorker(
        "cam-1",
        "facility-1",
        (),
        {},
        scheduler=Scheduler({}),
        domain_detectors=(_IdentitylessSequenceDetector(),),
        event_sink=sink,
    )

    worker.process_frame(_frame())

    assert len(sink.events) == 2
