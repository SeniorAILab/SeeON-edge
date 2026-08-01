"""CameraWorker audit/snapshot attachment migration (naming trap: this file
historically imported the legacy edge package despite its filename).

Disposition summary (full detail reported to the orchestrator alongside this
change):

- `OverlayRenderer.encode_jpeg_bounded` size-bounding and encode-failure
  behavior is unsuperseded and ported below, against
  worker/pipeline/output/overlay.py (byte-identical logic to
  edge/runtime/overlay_renderer.py, confirmed by direct comparison).
- The four `CameraWorker._attach_alert_metadata` tests (audit dict attach,
  snapshot bytes attach, audit-without-snapshot-renderer, non-alert-untouched)
  were IMPOSSIBLE to port at first: worker/pipeline/output/event_sink.py's
  `EvidenceEventSink.emit()` built a bare `EventPayload` and never called
  `audit_metadata_provider`/`encode_jpeg_bounded`/`SnapshotStore.store()`.
  Closed under todo 20 by `worker/pipeline/output/evidence_attacher.py`'s
  `AlertEvidenceAttacher` (composed by `CameraPipelinePump`, injected by
  `worker/runtime/worker.py`) plus `EvidenceEventSink.emit()` now persisting
  `event.snapshot_jpeg` through `SnapshotStore` when present. Ported below
  directly against `AlertEvidenceAttacher.attach` (the audit/snapshot
  attachment unit) and `EvidenceEventSink.emit` (the snapshot-storage unit),
  mirroring edge/runtime/camera_worker.py:289-336's semantics.
- The two `_RelayClient.emit()` payload-shape tests are superseded by
  tests/test_evidence_stager.py (DurableEvidenceStager._canonical_payload
  now owns that responsibility): `test_stager_commits_canonical_relay_payload_before_clip_binding`
  covers the with-audit-and-snapshot shape; the envelope-less negative shape
  now has a direct assertion in `test_evidence_stager.py` (residual coverage
  gap closed under todo 20) rather than resting on inspection alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import cv2
import numpy as np
import pytest

from contracts.event import EventPayload
from contracts.frame import Frame
from contracts.observation import FrameObservation
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.output.overlay import MAX_SNAPSHOT_BYTES, OverlayRenderer
from worker.types import BusinessEvent, FramePacket


def _packet(image: np.ndarray) -> FramePacket:
    return FramePacket(
        camera_id="camera-1",
        frame=Frame(index=1, time_sec=1.0, image=image),
        pts=1.0,
        seq=1,
        width=image.shape[1],
        height=image.shape[0],
        decode_time_ms=0.0,
    )


def _event(domain: str, event_type: str) -> BusinessEvent:
    return BusinessEvent(
        domain=domain,
        event_type=event_type,
        identity="event-123",
        camera_id="camera-1",
        facility_id="facility-1",
        time_sec=1.0,
        probability=0.87,
    )


@final
class _StubSnapshotRenderer:
    """Structural `SnapshotRenderer`: always returns fixed bytes."""

    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[object, ...] = (),
    ) -> bytes | None:
        del packet, observation, debug_snapshots
        return b"jpeg-bytes"


def test_overlay_renderer_encode_jpeg_bounded_returns_size_limited_bytes() -> None:
    jpeg = OverlayRenderer().encode_jpeg_bounded(
        _packet(np.zeros((480, 640, 3), dtype=np.uint8)),
        FrameObservation(),
    )

    assert jpeg is not None
    assert jpeg.startswith(b"\xff\xd8")
    assert len(jpeg) <= MAX_SNAPSHOT_BYTES


def test_overlay_renderer_encode_jpeg_bounded_returns_none_on_encode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> tuple[bool, np.ndarray]:
        del args, kwargs
        raise cv2.error("encode failed")

    monkeypatch.setattr(cv2, "imencode", fail)

    packet = _packet(np.zeros((64, 64, 3), dtype=np.uint8))
    assert OverlayRenderer().encode_jpeg_bounded(packet, FrameObservation()) is None


@pytest.mark.parametrize("domain", ["fall", "bed_exit"])
def test_alert_evidence_attacher_attaches_audit_and_snapshot_for_alert_events(
    domain: str,
) -> None:
    """Mirrors edge's `test_camera_worker_attaches_audit_and_snapshot_for_alert_events`
    against `AlertEvidenceAttacher.attach` directly (edge/runtime/camera_worker.py:289-336).
    """
    audit = {
        "clock_source": "edge_wall_clock",
        "model_version": "fall-model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.73,
    }
    attacher = AlertEvidenceAttacher(
        domain_audit={domain: audit},
        overlay_renderer=_StubSnapshotRenderer(),
    )
    packet = _packet(np.zeros((64, 64, 3), dtype=np.uint8))

    attached = attacher.attach(_event(domain, "tick"), packet, FrameObservation())

    assert attached.audit == audit
    assert attached.snapshot_jpeg == b"jpeg-bytes"


def test_alert_evidence_attacher_output_stores_the_rendered_snapshot_bytes(
    tmp_path: Path,
) -> None:
    """Mirrors edge's `test_camera_worker_stores_the_rendered_snapshot_bytes`: the
    attacher produces `snapshot_jpeg` bytes, and `EvidenceEventSink.emit` (the
    consumption side, worker/pipeline/output/event_sink.py) persists them
    through `SnapshotStore` and adds the `snapshot` record onto the payload.
    """
    attacher = AlertEvidenceAttacher(
        domain_audit={"fall": {"clock_source": "edge_wall_clock"}},
        overlay_renderer=_StubSnapshotRenderer(),
    )
    attached = attacher.attach(
        _event("fall", "tick"), _packet(np.zeros((64, 64, 3), dtype=np.uint8)), FrameObservation()
    )

    stager = _RecordingStager()
    sink = EvidenceEventSink(
        stager=stager,
        recorder=_RecordingRecorder(clip_id=None),
        snapshot_store=SnapshotStore(tmp_path),
        now=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    sink.emit(attached)

    payload = stager.staged[0]
    snapshot = payload["snapshot"]
    assert isinstance(snapshot, dict)
    path = tmp_path / str(snapshot["path"])
    assert path.read_bytes() == b"jpeg-bytes"
    assert snapshot["snapshot_id"] == payload["edge_event_id"]
    assert payload["snapshot_jpeg"] == b"jpeg-bytes"


def test_alert_evidence_attacher_attaches_audit_without_snapshot_renderer() -> None:
    """Mirrors edge's `test_camera_worker_attaches_audit_without_snapshot_renderer`:
    a missing renderer still attaches the audit envelope, just no snapshot.
    """
    attacher = AlertEvidenceAttacher(
        domain_audit={"fall": {"clock_source": "edge_wall_clock"}},
        overlay_renderer=None,
    )
    packet = _packet(np.zeros((64, 64, 3), dtype=np.uint8))

    attached = attacher.attach(_event("fall", "tick"), packet, FrameObservation())

    assert attached.audit == {"clock_source": "edge_wall_clock"}
    assert attached.snapshot_jpeg is None


def test_alert_evidence_attacher_leaves_non_alert_event_unaffected() -> None:
    """Mirrors edge's `test_camera_worker_leaves_non_alert_event_unaffected`: a
    domain with no registered audit envelope passes the event through untouched,
    even when a renderer is configured.
    """
    attacher = AlertEvidenceAttacher(
        domain_audit={"fall": {"clock_source": "edge_wall_clock"}},
        overlay_renderer=_StubSnapshotRenderer(),
    )
    packet = _packet(np.zeros((64, 64, 3), dtype=np.uint8))
    event = _event("movement-increase", "movement-increase")

    attached = attacher.attach(event, packet, FrameObservation())

    assert attached.audit is None
    assert attached.snapshot_jpeg is None
    assert attached is event


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
