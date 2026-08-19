from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from backend.app.features.evidence.explanation_schemas import (
    EventExplanationCorrelation,
    EventExplanationDelivery,
    EventExplanationMedia,
)
from shared.edge_db.migrator import migrate_database
from shared.edge_db.schema import SCHEMA_VERSION
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    EdgeEventId,
    EvidenceOutbox,
    StagedEvent,
)

NOW = "2026-08-15T00:00:15Z"
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
ANALYSIS_ID = hashlib.sha256(b"analysis-sections").hexdigest()
TRACE_ID = hashlib.sha256(b"trace-sections").hexdigest()
PAYLOAD_SENTINEL = "payload-json-SENTINEL-9f3c"
NOTES_SENTINEL = "operator-notes-SENTINEL-2ab1"
PATH_SENTINEL = "clips/private-SENTINEL-path/clip.mp4"
BYTES_SENTINEL = 424242
ERROR_SENTINEL = "Traceback: raw boom at /secret/path"


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _trace_ids(edge_event_id: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"analysis:{edge_event_id}".encode()).hexdigest(),
        hashlib.sha256(f"trace:{edge_event_id}".encode()).hexdigest(),
    )


def _seed_manifest(connection: sqlite3.Connection) -> None:
    existing = connection.execute(
        "SELECT 1 FROM runtime_manifest_contents WHERE manifest_sha256 = ?",
        (MANIFEST_ID,),
    ).fetchone()
    if existing is not None:
        return
    connection.execute(
        "INSERT INTO runtime_manifest_contents VALUES (?,1,'{}',?)",
        (MANIFEST_ID, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_boots VALUES ('boot-a',?,?)",
        (MANIFEST_ID, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_cameras VALUES ('boot-a','camera-a',?,?)",
        (MANIFEST_ID, NOW),
    )


def _next_frame_seq(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(frame_seq), -1) + 1 FROM runtime_analysis_traces "
        "WHERE worker_boot_id = 'boot-a' AND camera_id = 'camera-a' AND stream_epoch = 1"
    ).fetchone()
    return 0 if row is None else int(row[0])


def _seed_decision_for_event(connection: sqlite3.Connection, edge_event_id: str) -> str:
    analysis_id, trace_id = _trace_ids(edge_event_id)
    connection.execute(
        "INSERT INTO runtime_analysis_traces "
        "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,"
        "frame_seq,pts,source_time_sec,frame_width,frame_height,"
        "bed_region_provenance,storage_bytes) "
        "VALUES (?,1,'boot-a','camera-a',1,?,0,0,320,180,'fresh',1)",
        (analysis_id, _next_frame_seq(connection)),
    )
    connection.execute(
        "INSERT INTO evidence_decision_traces "
        "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,"
        "policy_qualified_id,effective_policy_id,runtime_manifest_sha256,reason,"
        "previous_state,current_state,triggered,track_id,track_missing_reason,"
        "bed_id,bed_missing_reason) "
        "VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset','clear','fall',1,"
        "7,NULL,NULL,'not-applicable')",
        (trace_id, analysis_id, POLICY_ID, MANIFEST_ID),
    )
    return trace_id


def _payload(edge_event_id: str) -> str:
    return json.dumps(
        {
            "camera_id": "camera-a",
            "event_type": "fall",
            "facility_id": "facility-private",
            "notes": NOTES_SENTINEL,
            "payload_json": PAYLOAD_SENTINEL,
            "edge_event_id": edge_event_id,
        },
        separators=(",", ":"),
    )


def _stage(
    database: Path,
    edge_event_id: str,
    *,
    queued_at: float,
    bind_clip: bool,
    clip_id: str | None = None,
) -> EdgeEventId:
    event_id = EdgeEventId(edge_event_id)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        trace_id = _seed_decision_for_event(connection, edge_event_id)
        connection.commit()
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=event_id,
                detected_at=NOW,
                payload_json=_payload(edge_event_id),
                queued_at=queued_at,
            ),
            required_runtime_manifest_sha256=MANIFEST_ID,
            required_decision_trace_id=trace_id,
        )
        if bind_clip:
            outbox.bind_clip(event_id, ClipId(clip_id or f"clip-{edge_event_id}"))
    return event_id


def _claim(outbox: EvidenceOutbox, *, now: float, owner: str = "sender-a"):
    claimed = outbox.claim(ClaimLease(owner=owner, now=now, duration=10.0))
    assert claimed is not None
    return claimed


def _event_row(database: Path, edge_event_id: str) -> tuple[object, ...]:
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT edge_event_id, state, delivery_state, attempt_count, "
            "backend_event_id, last_error_code "
            "FROM evidence_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone()
    assert row is not None
    return row


def _project(database: Path, edge_event_id: str, **kwargs: Any):
    from backend.app.features.evidence.explanation_sections import (
        project_explanation_sections,
    )

    return project_explanation_sections(database, edge_event_id, **kwargs)


def _dump(section: Any) -> dict[str, Any]:
    if hasattr(section, "model_dump"):
        return section.model_dump(mode="json")
    return {
        "delivery": section.delivery.model_dump(mode="json"),
        "media": section.media.model_dump(mode="json"),
        "correlation": section.correlation.model_dump(mode="json"),
    }


def test_outbox_delivery_attempt_semantics_keep_one_event(tmp_path: Path) -> None:
    # Given: one schema-v16 event that the sender claims three times.
    assert SCHEMA_VERSION == 16
    database = _database(tmp_path)
    event_id = "event:attempts"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with sqlite3.connect(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info('evidence_events')")
        }
    assert version == (16,)
    assert "last_http_status" not in columns
    assert "alert_id" not in columns

    with EvidenceOutbox.open(database) as outbox:
        first = _claim(outbox, now=100.0)
        assert outbox.schedule_retry(first, next_attempt_at=110.0)
        second = _claim(outbox, now=110.0)
        assert outbox.schedule_retry(second, next_attempt_at=120.0)
        third = _claim(outbox, now=120.0)
        assert third.edge_event_id == event_id
        assert third.attempt_count == 3

    # When: the durable outbox row is read back.
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT COUNT(*), MAX(attempt_count), MAX(delivery_state) "
            "FROM evidence_events WHERE edge_event_id = ?",
            (event_id,),
        ).fetchone()

    # Then: three attempts remain one PENDING event and HTTP status is unpersisted.
    assert rows == (1, 3, "PENDING")
    assert _event_row(database, event_id)[0] == event_id


def test_outbox_delivery_ack_pending_terminal_and_local_states_are_distinct(
    tmp_path: Path,
) -> None:
    # Given: five independently staged schema-v16 events.
    database = _database(tmp_path)
    local_id = "event:local"
    never_id = "event:never"
    pending_id = "event:pending"
    ack_id = "event:ack"
    terminal_id = "event:terminal"
    _stage(database, local_id, queued_at=1.0, bind_clip=False)
    _stage(database, never_id, queued_at=2.0, bind_clip=False)
    _stage(database, pending_id, queued_at=3.0, bind_clip=True)
    _stage(database, ack_id, queued_at=4.0, bind_clip=True)
    _stage(database, terminal_id, queued_at=5.0, bind_clip=True)

    with EvidenceOutbox.open(database) as outbox:
        pending = _claim(outbox, now=100.0)
        assert pending.edge_event_id == pending_id
        assert outbox.schedule_retry(pending, next_attempt_at=200.0)
        acked = outbox.claim(ClaimLease("sender-a", 100.0, 10.0))
        assert acked is not None and acked.edge_event_id == ack_id
        assert outbox.acknowledge(acked, backend_event_id="backend-ack")
        failed = outbox.claim(ClaimLease("sender-a", 100.0, 10.0))
        assert failed is not None and failed.edge_event_id == terminal_id
        assert outbox.mark_event_failure(failed, state="PERMANENT", error_code="HTTP_404")
        assert outbox.mark_ready(EdgeEventId(never_id))

    kinds: dict[str, tuple[str, str, int, str | None, str | None]] = {}
    with sqlite3.connect(database) as connection:
        for event_id in (local_id, never_id, pending_id, ack_id, terminal_id):
            row = connection.execute(
                "SELECT state, delivery_state, attempt_count, backend_event_id, "
                "last_error_code FROM evidence_events WHERE edge_event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            kinds[event_id] = (
                str(row[0]),
                str(row[1]),
                int(row[2]),
                None if row[3] is None else str(row[3]),
                None if row[4] is None else str(row[4]),
            )

    # Then: the five durable combinations stay distinct and never store HTTP status.
    assert kinds[local_id] == ("STAGED", "PENDING", 0, None, None)
    assert kinds[never_id] == ("READY", "PENDING", 0, None, None)
    assert kinds[pending_id] == ("READY", "PENDING", 1, None, None)
    assert kinds[ack_id] == ("ACKED", "ACKED", 1, "backend-ack", None)
    assert kinds[terminal_id] == ("READY", "PERMANENT", 1, None, "HTTP_404")
    assert len(set(kinds.values())) == 5


def test_delivery_ack_projects_one_event_and_typed_http_unavailability(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    event_id = "event:ack-project"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert outbox.acknowledge(claimed, backend_event_id="backend-ack")

    delivery = _project(database, event_id).delivery
    assert isinstance(delivery, EventExplanationDelivery)
    assert delivery.status == "COMPLETE"
    assert delivery.reasons == []
    assert delivery.outbox_state.value == "ACKED"
    assert delivery.attempt_count.value == 1
    assert delivery.last_delivery_disposition.value is None
    assert delivery.last_delivery_disposition.missing_reason == "disposition_not_persisted"
    assert delivery.last_http_status.value is None
    assert delivery.last_http_status.missing_reason == "last_http_status_not_persisted"
    assert delivery.backend_event_id.value == "backend-ack"


def _delivery_key(delivery: EventExplanationDelivery) -> tuple[object, ...]:
    return (
        delivery.outbox_state.value,
        delivery.outbox_state.missing_reason,
        delivery.attempt_count.value,
        delivery.last_delivery_disposition.value,
        delivery.last_delivery_disposition.missing_reason,
        delivery.backend_event_id.value,
        delivery.backend_event_id.missing_reason,
    )


def test_delivery_pending_and_never_attempted_and_local_only_are_distinct(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    local_id = "event:local-project"
    never_id = "event:never-project"
    pending_id = "event:pending-project"
    ack_id = "event:ack-distinct"
    terminal_id = "event:terminal-distinct"
    _stage(database, local_id, queued_at=1.0, bind_clip=False)
    _stage(database, never_id, queued_at=2.0, bind_clip=False)
    _stage(database, pending_id, queued_at=3.0, bind_clip=True)
    _stage(database, ack_id, queued_at=4.0, bind_clip=True)
    _stage(database, terminal_id, queued_at=5.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert claimed.edge_event_id == pending_id
        assert outbox.schedule_retry(claimed, next_attempt_at=200.0)
        acked = outbox.claim(ClaimLease("sender-a", 100.0, 10.0))
        assert acked is not None and acked.edge_event_id == ack_id
        assert outbox.acknowledge(acked, backend_event_id="backend-ack")
        failed = outbox.claim(ClaimLease("sender-a", 100.0, 10.0))
        assert failed is not None and failed.edge_event_id == terminal_id
        assert outbox.mark_event_failure(failed, state="PERMANENT", error_code="HTTP_404")
        assert outbox.mark_ready(EdgeEventId(never_id))

    local = _project(database, local_id).delivery
    never = _project(database, never_id).delivery
    pending = _project(database, pending_id).delivery
    acked_delivery = _project(database, ack_id).delivery
    terminal = _project(database, terminal_id).delivery

    assert local.outbox_state.value is None
    assert local.outbox_state.missing_reason == "delivery_never_attempted"
    assert local.attempt_count.value == 0
    assert local.last_delivery_disposition.missing_reason == "delivery_never_attempted"
    assert never.outbox_state.value == "PENDING"
    assert never.attempt_count.value == 0
    assert never.last_delivery_disposition.missing_reason == "delivery_never_attempted"
    assert pending.outbox_state.value == "PENDING"
    assert pending.attempt_count.value == 1
    assert pending.last_delivery_disposition.missing_reason == "disposition_not_persisted"
    assert pending.backend_event_id.missing_reason == "backend_event_id_not_persisted"
    assert acked_delivery.outbox_state.value == "ACKED"
    assert terminal.outbox_state.value == "PERMANENT"
    assert _delivery_key(local) != _delivery_key(never)
    keys = {
        _delivery_key(local),
        _delivery_key(never),
        _delivery_key(pending),
        _delivery_key(acked_delivery),
        _delivery_key(terminal),
    }
    assert len(keys) == 5
    assert never.status == "COMPLETE"
    assert pending.status == "COMPLETE"
    assert acked_delivery.status == "COMPLETE"
    assert terminal.status == "COMPLETE"


def test_delivery_terminal_disposition_is_exact_and_http_stays_unpersisted(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    event_id = "event:terminal-project"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert outbox.mark_event_failure(claimed, state="PERMANENT", error_code="HTTP_404")

    delivery = _project(database, event_id).delivery
    assert delivery.outbox_state.value == "PERMANENT"
    assert delivery.last_delivery_disposition.value == "PERMANENT"
    assert delivery.last_http_status.value is None
    assert delivery.last_http_status.missing_reason == "last_http_status_not_persisted"
    assert delivery.last_http_status.value != 404
    assert delivery.last_http_status.value != 0
    assert delivery.status == "COMPLETE"


def test_delivery_compatibility_terminal_is_exact(tmp_path: Path) -> None:
    database = _database(tmp_path)
    event_id = "event:compat-project"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert outbox.mark_event_failure(
            claimed,
            state="COMPATIBILITY",
            error_code="HTTP_404",
        )

    delivery = _project(database, event_id).delivery
    assert delivery.outbox_state.value == "COMPATIBILITY"
    assert delivery.last_delivery_disposition.value == "COMPATIBILITY"
    assert delivery.last_http_status.missing_reason == "last_http_status_not_persisted"


def test_delivery_multiple_attempts_remain_one_event_without_duplicate_conclusion(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    event_id = "event:three-attempts"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        first = _claim(outbox, now=100.0)
        assert outbox.schedule_retry(first, next_attempt_at=110.0)
        second = _claim(outbox, now=110.0)
        assert outbox.schedule_retry(second, next_attempt_at=120.0)
        third = _claim(outbox, now=120.0)
        assert outbox.acknowledge(third, backend_event_id="backend-three")

    sections = _project(database, event_id)
    dumped = json.dumps(_dump(sections), separators=(",", ":"))
    assert isinstance(sections.delivery, EventExplanationDelivery)
    assert sections.delivery.attempt_count.value == 3
    assert sections.delivery.outbox_state.value == "ACKED"
    assert sections.delivery.backend_event_id.value == "backend-three"
    assert dumped.count(event_id) == 0 or sections.delivery.attempt_count.value == 3
    assert "DELIVERY_RETRY" not in dumped
    assert "duplicate" not in dumped.lower()
    assert "unique_edge_event_count" not in dumped


def test_delivery_missing_backend_id_is_unavailable_not_empty(tmp_path: Path) -> None:
    database = _database(tmp_path)
    event_id = "event:missing-backend"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert outbox.acknowledge(claimed)

    delivery = _project(database, event_id).delivery
    assert delivery.outbox_state.value == "ACKED"
    assert delivery.backend_event_id.value is None
    assert delivery.backend_event_id.missing_reason == "backend_event_id_not_persisted"
    assert delivery.status == "COMPLETE"


def test_media_snapshot_and_clip_present_absent_and_retained(tmp_path: Path) -> None:
    database = _database(tmp_path)
    present_id = "event:media-present"
    absent_id = "event:media-absent"
    retained_id = "event:media-retained"
    _stage(database, present_id, queued_at=1.0, bind_clip=True, clip_id="clip-present")
    _stage(database, absent_id, queued_at=2.0, bind_clip=False)
    _stage(database, retained_id, queued_at=3.0, bind_clip=True, clip_id="clip-retained")
    with EvidenceOutbox.open(database) as outbox:
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=ClipId("clip-present"),
                local_state=ClipLocalState.VERIFIED,
                manifest_path="clips/present-SENTINEL-path/manifest.json",
                state_version=2,
                media_relpath="clips/present-SENTINEL-path/clip.mp4",
                sha256="b" * 64,
                size_bytes=BYTES_SENTINEL,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at=NOW,
                clip_end_at=NOW,
                finalized_at=NOW,
                manifest_sha256="1" * 64,
                manifest_size_bytes=12,
            )
        )
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=ClipId("clip-retained"),
                local_state=ClipLocalState.VERIFIED,
                manifest_path="clips/retained-SENTINEL-path/manifest.json",
                state_version=2,
                media_relpath="clips/retained-SENTINEL-path/clip.mp4",
                sha256="c" * 64,
                size_bytes=BYTES_SENTINEL,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at=NOW,
                clip_end_at=NOW,
                finalized_at=NOW,
                manifest_sha256="2" * 64,
                manifest_size_bytes=12,
            )
        )
        claimed = outbox.claim(ClaimLease("sender-a", 100.0, 10.0))
        assert claimed is not None and claimed.edge_event_id == present_id
        assert outbox.acknowledge(claimed, backend_event_id="backend-present")
        published = outbox.claim_clip(ClaimLease("clip-sender", 100.0, 10.0))
        assert published is not None and published.clip_id == ClipId("clip-present")
        assert outbox.acknowledge_clip(
            published,
            acknowledged_at=100.0,
            remote_state="READY",
        )

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO evidence_media_objects (
                media_id, content_sha256, size_bytes, mime_type,
                contained_relpath, basename, created_at
            ) VALUES ('media:snap', ?, 12, 'image/jpeg', ?, 'snap.jpg', ?)
            """,
            ("d" * 64, PATH_SENTINEL, NOW),
        )
        connection.execute(
            "UPDATE evidence_artifact_slots SET state = 'AVAILABLE', "
            "media_id = 'media:snap', reason = NULL, revision = revision + 1, "
            "updated_at = ? WHERE incident_id = ? AND slot_name = 'SNAPSHOT'",
            (NOW, present_id),
        )
        connection.execute(
            """
            INSERT INTO derivative_evidence_slots (
                incident_id, derivative_kind, state, media_id, created_at, updated_at
            ) VALUES (?, 'ANNOTATED_CLIP', 'AVAILABLE', 'media:snap', ?, ?)
            """,
            (present_id, NOW, NOW),
        )
        connection.execute(
            """
            INSERT INTO evidence_retention_states (
                clip_id, state, requested_at, updated_at
            ) VALUES ('clip-retained', 'PURGED', ?, ?)
            """,
            (NOW, NOW),
        )
        connection.execute(
            "UPDATE evidence_artifact_slots SET state = 'UNAVAILABLE', "
            "reason = 'RETENTION_PURGED', revision = revision + 1, "
            "updated_at = ? WHERE incident_id = ? AND slot_name = 'PRIMARY_CLIP'",
            (NOW, retained_id),
        )
        connection.commit()

    present = _project(database, present_id).media
    absent = _project(database, absent_id).media
    retained = _project(database, retained_id).media

    assert isinstance(present, EventExplanationMedia)
    assert present.status == "COMPLETE"
    assert present.snapshot.state == "AVAILABLE"
    assert present.clip.state == "AVAILABLE"
    assert present.snapshot.missing_reason is None
    assert present.clip.missing_reason is None

    assert absent.snapshot.state == "UNAVAILABLE"
    assert absent.snapshot.missing_reason == "snapshot_not_recorded"
    assert absent.clip.state == "NOT_RECORDED"
    assert absent.clip.missing_reason == "clip_not_recorded"
    assert absent.status == "UNAVAILABLE"

    assert retained.clip.state == "UNAVAILABLE"
    assert retained.clip.missing_reason == "artifact_unavailable"
    dumped = json.dumps(_dump(present), separators=(",", ":"))
    assert PATH_SENTINEL not in dumped
    assert str(BYTES_SENTINEL) not in dumped
    assert "media_relpath" not in dumped
    assert "size_bytes" not in dumped


def test_media_unpublished_verified_waiting_and_derivative_are_not_available(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    unpublished_id = "event:media-unpublished"
    derivative_only_id = "event:media-derivative-only"
    _stage(database, unpublished_id, queued_at=1.0, bind_clip=True, clip_id="clip-waiting")
    _stage(
        database,
        derivative_only_id,
        queued_at=2.0,
        bind_clip=True,
        clip_id="clip-derivative",
    )
    with EvidenceOutbox.open(database) as outbox:
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=ClipId("clip-waiting"),
                local_state=ClipLocalState.VERIFIED,
                manifest_path="clips/waiting-SENTINEL-path/manifest.json",
                state_version=2,
                media_relpath="clips/waiting-SENTINEL-path/clip.mp4",
                sha256="b" * 64,
                size_bytes=BYTES_SENTINEL,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at=NOW,
                clip_end_at=NOW,
                finalized_at=NOW,
                manifest_sha256="1" * 64,
                manifest_size_bytes=12,
            )
        )
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=ClipId("clip-derivative"),
                local_state=ClipLocalState.VERIFIED,
                manifest_path="clips/derivative-SENTINEL-path/manifest.json",
                state_version=2,
                media_relpath="clips/derivative-SENTINEL-path/clip.mp4",
                sha256="c" * 64,
                size_bytes=BYTES_SENTINEL,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at=NOW,
                clip_end_at=NOW,
                finalized_at=NOW,
                manifest_sha256="2" * 64,
                manifest_size_bytes=12,
            )
        )

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        waiting = connection.execute(
            "SELECT local_state, publish_state FROM evidence_clips WHERE clip_id = ?",
            ("clip-waiting",),
        ).fetchone()
        assert waiting == ("VERIFIED", "WAITING")
        connection.execute(
            """
            INSERT INTO evidence_media_objects (
                media_id, content_sha256, size_bytes, mime_type,
                contained_relpath, basename, created_at
            ) VALUES ('media:derivative', ?, 12, 'video/mp4', ?, 'annotated.mp4', ?)
            """,
            ("d" * 64, PATH_SENTINEL, NOW),
        )
        connection.execute(
            """
            INSERT INTO derivative_evidence_slots (
                incident_id, derivative_kind, state, media_id, created_at, updated_at
            ) VALUES (?, 'ANNOTATED_CLIP', 'AVAILABLE', 'media:derivative', ?, ?)
            """,
            (derivative_only_id, NOW, NOW),
        )
        connection.execute(
            """
            INSERT INTO derivative_evidence_slots (
                incident_id, derivative_kind, state, reason, created_at, updated_at
            ) VALUES (?, 'ANNOTATED_CLIP', 'UNAVAILABLE', 'ENCODER_FAILED', ?, ?)
            """,
            (unpublished_id, NOW, NOW),
        )
        connection.commit()

    unpublished = _project(database, unpublished_id).media
    derivative_only = _project(database, derivative_only_id).media

    assert unpublished.clip.state != "AVAILABLE"
    assert unpublished.clip.state == "PENDING"
    assert unpublished.clip.missing_reason is None
    assert unpublished.status != "COMPLETE"
    assert "artifact_unavailable" in unpublished.reasons
    assert derivative_only.clip.state != "AVAILABLE"
    assert derivative_only.clip.state == "PENDING"
    dumped = json.dumps(_dump(unpublished), separators=(",", ":"))
    assert PATH_SENTINEL not in dumped
    assert str(BYTES_SENTINEL) not in dumped


def test_correlation_is_out_of_repo_unless_allowlisted_export_matches(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    event_id = "event:correlation"
    _stage(database, event_id, queued_at=100.0, bind_clip=True)
    with EvidenceOutbox.open(database) as outbox:
        claimed = _claim(outbox, now=100.0)
        assert outbox.acknowledge(claimed, backend_event_id="backend-corr")

    from backend.app.features.evidence.explanation_sections import (
        AlertCorrelationExport,
    )

    without_export = _project(database, event_id).correlation
    assert isinstance(without_export, EventExplanationCorrelation)
    assert without_export.status == "UNAVAILABLE"
    assert without_export.reasons == ["alert_correlation_export_not_supplied"]
    assert without_export.alert_id.value is None
    assert without_export.alert_id.missing_reason == "alert_correlation_export_not_supplied"

    mismatched = _project(
        database,
        event_id,
        alert_correlation_export=AlertCorrelationExport(
            edge_event_id="event:other",
            alert_id="alert-other",
        ),
    ).correlation
    assert mismatched.status == "UNAVAILABLE"
    assert mismatched.alert_id.missing_reason == "alert_correlation_export_not_supplied"
    assert mismatched.alert_id.value is None

    matched = _project(
        database,
        event_id,
        alert_correlation_export=AlertCorrelationExport(
            edge_event_id=event_id,
            alert_id="alert-1",
        ),
    ).correlation
    assert matched.status == "COMPLETE"
    assert matched.reasons == []
    assert matched.alert_id.value == "alert-1"


def test_delivery_media_correlation_privacy_sentinels_are_absent(tmp_path: Path) -> None:
    database = _database(tmp_path)
    event_id = "event:privacy"
    _stage(database, event_id, queued_at=100.0, bind_clip=True, clip_id="clip-privacy")
    with EvidenceOutbox.open(database) as outbox:
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=ClipId("clip-privacy"),
                local_state=ClipLocalState.VERIFIED,
                manifest_path=PATH_SENTINEL,
                state_version=2,
                media_relpath=PATH_SENTINEL,
                sha256="e" * 64,
                size_bytes=BYTES_SENTINEL,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at=NOW,
                clip_end_at=NOW,
                finalized_at=NOW,
                manifest_sha256="3" * 64,
                manifest_size_bytes=12,
            )
        )
        claimed = _claim(outbox, now=100.0)
        assert outbox.acknowledge(claimed, backend_event_id="backend-private")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET last_error_code = ? WHERE edge_event_id = ?",
            (ERROR_SENTINEL, event_id),
        )
        connection.commit()

    dumped = json.dumps(_dump(_project(database, event_id)), separators=(",", ":"))
    assert PAYLOAD_SENTINEL not in dumped
    assert NOTES_SENTINEL not in dumped
    assert PATH_SENTINEL not in dumped
    assert ERROR_SENTINEL not in dumped
    assert str(BYTES_SENTINEL) not in dumped
    assert "payload_json" not in dumped
    assert "media_relpath" not in dumped
    assert "contained_relpath" not in dumped
    assert "size_bytes" not in dumped
    assert "notes" not in dumped
    assert "last_error_code" not in dumped


def test_malformed_input_is_rejected_without_projection(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _stage(database, "event:valid", queued_at=1.0, bind_clip=False)
    from backend.app.features.evidence.explanation_sections import (
        project_explanation_sections,
    )

    with pytest.raises(ValueError):
        project_explanation_sections(database, "")
    with pytest.raises(ValueError):
        project_explanation_sections(database, "bad\x00id")
    missing = project_explanation_sections(database, "event:missing")
    assert missing.delivery.status == "UNAVAILABLE"
    assert missing.delivery.reasons == ["outbox_row_unresolved"]
    assert missing.delivery.outbox_state.missing_reason == "outbox_row_unresolved"
    assert missing.media.status == "UNAVAILABLE"
    assert missing.media.reasons == ["clip_not_recorded"]
    assert missing.correlation.status == "UNAVAILABLE"
    assert missing.correlation.reasons == ["alert_correlation_export_not_supplied"]
