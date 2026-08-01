from __future__ import annotations

import base64
import json
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimLease,
    ClipId,
    EdgeEventId,
    EvidenceOutbox,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

EVENT_ID = EdgeEventId("00000000-0000-4000-8000-000000000001")


def _event() -> dict[str, object]:
    return {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.93,
        "detected_at": "2026-07-16T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "audit": {"model_version": "model-1"},
        "snapshot_jpeg": b"jpeg-bytes",
        "snapshot": {
            "snapshot_id": "event-1",
            "path": "snapshots/camera/2026-07-16/event-1.jpg",
            "sha256": "abc123",
            "size_bytes": 10,
            "mime_type": "image/jpeg",
            "captured_at": "2026-07-16T00:00:00Z",
            "camera_id": "camera-1",
            "edge_event_id": EVENT_ID,
        },
    }


def test_stager_commits_canonical_relay_payload_before_clip_binding(tmp_path: Path) -> None:
    database = tmp_path / "evidence.sqlite3"
    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id="resident-1",
        config_version=7,
        clock=lambda: 100.0,
    )

    stager.stage(_event())
    stager.complete(EVENT_ID, ClipId("clip-1"))

    with EvidenceOutbox.open(database) as outbox:
        claim = outbox.claim(ClaimLease("sender", 100.0, 10.0))
        assert outbox.ordered_event_ids(ClipId("clip-1")) == (EVENT_ID,)
    assert claim is not None
    payload = json.loads(claim.payload_json)
    assert payload == {
        "audit": {"config_version": 7, "model_version": "model-1"},
        "camera_id": "camera-1",
        "detected_at": "2026-07-16T00:00:00Z",
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "evidence": {
            "camera_id": "camera-1",
            "detected_at": "2026-07-16T00:00:00Z",
            "edge_event_id": EVENT_ID,
            "event_type": "fall",
            "facility_id": "facility-1",
            "probability": 0.93,
        },
        "facility_id": "facility-1",
        "probability": 0.93,
        "resident_id": "resident-1",
        "snapshot": {
            "camera_id": "camera-1",
            "captured_at": "2026-07-16T00:00:00Z",
            "edge_event_id": EVENT_ID,
            "mime_type": "image/jpeg",
            "path": "snapshots/camera/2026-07-16/event-1.jpg",
            "sha256": "abc123",
            "size_bytes": 10,
            "snapshot_id": "event-1",
        },
        "snapshot_jpeg_base64": base64.b64encode(b"jpeg-bytes").decode("ascii"),
    }
    assert str(database) not in claim.payload_json


def _envelope_less_event() -> dict[str, object]:
    """A staged event with no `audit`/`snapshot_jpeg`/`snapshot` keys at all --
    the pre-todo-20 shape every non-alert event (and every alert event before
    `AlertEvidenceAttacher` existed) still has.
    """
    event = _event()
    del event["audit"]
    del event["snapshot_jpeg"]
    del event["snapshot"]
    return event


def test_stager_commits_envelope_less_payload_without_audit_or_snapshot_keys(
    tmp_path: Path,
) -> None:
    """`_canonical_payload`'s negative branch (worker/pipeline/output/evidence/
    evidence_stager.py:64-81): when `audit`/`snapshot_jpeg`/`snapshot` are all
    absent from the staged event, none of `audit`/`snapshot_jpeg_base64`/
    `snapshot` are added to the relay payload, and `evidence` carries no trace
    of them either (residual coverage closed under todo 20 -- previously only
    inferred by inspection of the `isinstance` guards, not asserted).
    """
    database = tmp_path / "evidence.sqlite3"
    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=7,
        clock=lambda: 100.0,
    )

    stager.stage(_envelope_less_event())
    stager.complete(EVENT_ID, None)

    with EvidenceOutbox.open(database) as outbox:
        claim = outbox.claim(ClaimLease("sender", 100.0, 10.0))
    assert claim is not None
    payload = json.loads(claim.payload_json)
    assert "audit" not in payload
    assert "snapshot_jpeg_base64" not in payload
    assert "snapshot" not in payload
    assert "audit" not in payload["evidence"]
    assert "snapshot_jpeg" not in payload["evidence"]
    assert "snapshot" not in payload["evidence"]


def test_stager_makes_event_without_clip_sendable(tmp_path: Path) -> None:
    database = tmp_path / "evidence.sqlite3"
    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=0,
        clock=lambda: 100.0,
    )
    stager.stage(_event())
    stager.complete(EVENT_ID, None)

    with EvidenceOutbox.open(database) as outbox:
        claim = outbox.claim(ClaimLease("sender", 100.0, 10.0))
    assert claim is not None
    assert claim.edge_event_id == EVENT_ID
