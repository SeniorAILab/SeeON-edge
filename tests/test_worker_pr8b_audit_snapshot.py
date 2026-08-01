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
  are IMPOSSIBLE to port: worker/pipeline/output/event_sink.py's
  `EvidenceEventSink.emit()` builds a bare `EventPayload` and never calls
  `audit_metadata_provider`/`encode_jpeg_bounded`/`SnapshotStore.store()`.
  Every individual building block exists worker-side (worker/domains/registry.py's
  `audit_metadata_provider`/`AuditContext`/`DomainAuditSnapshot`; this file's
  `OverlayRenderer.encode_jpeg_bounded`; worker/pipeline/output/evidence/
  snapshot_store.py's `SnapshotStore`), but nothing in the pump/decision/sink
  pipeline wires them together -- a genuine production feature gap, reported
  to the orchestrator rather than worked around here.
- The two `_RelayClient.emit()` payload-shape tests are superseded by
  tests/test_evidence_stager.py (DurableEvidenceStager._canonical_payload
  now owns that responsibility): `test_stager_commits_canonical_relay_payload_before_clip_binding`
  covers the with-audit-and-snapshot shape; the envelope-less negative shape
  is not separately asserted there but is trivially true by inspection of
  `_canonical_payload`'s `isinstance(event_audit, Mapping)` /
  `isinstance(snapshot_jpeg, bytes)` guards (worker/pipeline/output/evidence/
  evidence_stager.py:64-81), which only add those keys when present.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import FrameObservation
from worker.pipeline.output.overlay import MAX_SNAPSHOT_BYTES, OverlayRenderer
from worker.types import FramePacket


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
