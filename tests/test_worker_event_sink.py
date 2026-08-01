from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from contracts.event import EventPayload
from worker.interfaces.output import EventSink
from worker.types.business_event import BusinessEvent


@dataclass(slots=True)
class _RecordingStager:
    staged: list[EventPayload] = field(default_factory=list)
    completions: list[tuple[str, str | None]] = field(default_factory=list)

    def stage(self, event: EventPayload) -> None:
        self.staged.append(event)

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.completions.append((edge_event_id, clip_id))


@dataclass(slots=True)
class _RecordingRecorder:
    clip_id: str | None
    calls: list[tuple[str, str, str | None, bool]] = field(default_factory=list)

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        self.calls.append((camera_id, event_ref, event_type, allow_new_clip))
        return self.clip_id


def _event() -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall_detected",
        identity="event-123",
        camera_id="camera-1",
        facility_id="facility-1",
        time_sec=12.5,
        probability=0.91,
        person_id=7,
    )


def test_event_sink_stages_then_binds_the_admitted_business_event() -> None:
    from worker.pipeline.output.event_sink import EvidenceEventSink

    # Given: a durable stager and a recorder that reserves one clip.
    stager = _RecordingStager()
    recorder = _RecordingRecorder(clip_id="clip-123")
    sink = EvidenceEventSink(
        stager=stager,
        recorder=recorder,
        now=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    # When: the decision pipeline emits an admitted immutable event.
    sink.emit(_event())

    # Then: its canonical relay payload is durable before its clip relation completes.
    assert stager.staged == [
        {
            "edge_event_id": "event-123",
            "event_type": "fall_detected",
            "probability": 0.91,
            "detected_at": "2026-07-31T12:00:00Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "evidence": {
                "domain": "fall",
                "identity": "event-123",
                "time_sec": 12.5,
                "person_id": 7,
            },
        }
    ]
    assert recorder.calls == [("camera-1", "event-123", "fall_detected", True)]
    assert stager.completions == [("event-123", "clip-123")]
    assert isinstance(sink, EventSink)


def test_event_sink_completes_without_clip_when_recording_is_unavailable() -> None:
    from worker.pipeline.output.event_sink import EvidenceEventSink

    # Given: a recorder that cannot reserve a new clip.
    stager = _RecordingStager()
    sink = EvidenceEventSink(
        stager=stager,
        recorder=_RecordingRecorder(clip_id=None),
        now=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    # When: the event is emitted.
    sink.emit(_event())

    # Then: durable delivery remains ready rather than being dropped.
    assert stager.completions == [("event-123", None)]


def test_worker_relay_surface_delegates_http_to_the_shared_bounded_transport() -> None:
    # Given: the worker surfaces that emit facts and status to the backend.
    repo_root = Path(__file__).resolve().parents[1]
    relay_sources = (
        "worker/runtime/worker.py",
        "worker/runtime/telemetry/runtime_status_sender.py",
    )

    for relative in relay_sources:
        source = (repo_root / relative).read_text(encoding="utf-8")

        # When: that surface sends a request.
        # Then: it delegates to the one shared bounded transport, never opens HTTP itself.
        assert "bounded_request" in source, relative
        assert "urllib.request.urlopen" not in source, relative
