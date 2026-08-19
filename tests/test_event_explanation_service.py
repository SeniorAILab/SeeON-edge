"""Characterization of Foundation seams for event-explanation composition."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from backend.app.features.evidence.explanation_manifest import project_runtime_manifest
from backend.app.features.evidence.explanation_neighborhood import (
    EXPECTED_NEIGHBORHOOD_FRAMES,
    EventNeighborhoodQuery,
)
from backend.app.features.evidence.explanation_schemas import EventExplanationResponse
from backend.app.features.evidence.explanation_sections import project_explanation_sections
from backend.app.features.evidence.explanation_service import (
    EventExplanationNotFound,
    EventExplanationService,
)
from backend.app.features.evidence.explanation_store import (
    EventExplanationFacts,
    EventExplanationQuery,
    TraceRefConflict,
)
from shared.edge_db.migrator import migrate_database

NOW = "2026-08-15T00:00:00Z"
CAMERA_ID = "camera-a"
BOOT_ID = "boot-a"
POLICY_ID = "b" * 64
ANALYSIS_A = hashlib.sha256(b"analysis-a").hexdigest()
TRACE_A = hashlib.sha256(b"trace-a").hexdigest()
TRACE_B = hashlib.sha256(b"trace-b").hexdigest()
TRIGGER_SEQ = 40
PRECEDING_FRAMES = 29
PRIVACY_SENTINEL = "PRIVACY_SENTINEL_service_9f3c21ab"
PATH_SENTINEL = "/private/media-path-sentinel.mp4"
GEOMETRY_SENTINEL = "[[987654,123456]]"
ACTOR_SENTINEL = "actor:sentinel-service-7c21e9aa"
NOTES_SENTINEL = "NOTES_SENTINEL_service_2ab17c21"
HISTORY_SENTINEL = "HISTORY_SENTINEL_service_revision_text_44e1"


def _canonical(content: object) -> tuple[str, str]:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def _current_manifest() -> tuple[str, str]:
    return _canonical(
        {
            "manifest_schema_version": 1,
            "build": {
                "detector_version": "detector-v1",
                "worker_build_revision": "a" * 40,
                "image_revision": "b" * 40,
            },
            "configuration": {"config_version": 3},
            "modules": [
                {
                    "qualified_id": "fall.v1",
                    "policy_schema": "fall.policy.v1",
                    "component_bindings": [
                        {"component_id": "fall-classifier", "kind": "model"}
                    ],
                }
            ],
            "components": [
                {
                    "component_id": "fall-classifier",
                    "artifact_sha256": "c" * 64,
                }
            ],
        }
    )


def _migrated(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _seed_manifest(
    connection: sqlite3.Connection,
    *,
    canonical_json: str,
    manifest_sha: str,
) -> None:
    connection.execute(
        "INSERT INTO runtime_manifest_contents VALUES (?,1,?,?)",
        (manifest_sha, canonical_json, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_boots VALUES (?,?,?)",
        (BOOT_ID, manifest_sha, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_cameras VALUES (?,?,?,?)",
        (BOOT_ID, CAMERA_ID, manifest_sha, NOW),
    )


def _insert_analysis(
    connection: sqlite3.Connection,
    *,
    analysis_id: str,
    frame_seq: int,
) -> None:
    connection.execute(
        "INSERT INTO runtime_analysis_traces "
        "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,"
        "frame_seq,pts,source_time_sec,frame_width,frame_height,"
        "bed_region_provenance,storage_bytes) "
        "VALUES (?,1,?,?,1,?,0,0,320,180,'fresh',1)",
        (analysis_id, BOOT_ID, CAMERA_ID, frame_seq),
    )


def _insert_decision(
    connection: sqlite3.Connection,
    *,
    trace_id: str,
    analysis_id: str | None,
    manifest_sha: str,
    values: tuple[tuple[str, float | None, str | None], ...] = (
        ("fall_probability", 0.91, None),
        ("operating_threshold", 0.5, None),
        ("containment_ratio", None, "not-applicable"),
    ),
) -> None:
    connection.execute(
        "INSERT INTO evidence_decision_traces "
        "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,"
        "policy_qualified_id,effective_policy_id,runtime_manifest_sha256,reason,"
        "previous_state,current_state,triggered,track_id,track_missing_reason,"
        "bed_id,bed_missing_reason) "
        "VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset','clear','fall',1,"
        "7,NULL,NULL,'not-applicable')",
        (trace_id, analysis_id, POLICY_ID, manifest_sha),
    )
    connection.executemany(
        "INSERT INTO evidence_decision_values "
        "(decision_trace_id, name, numeric_value, missing_reason) VALUES (?,?,?,?)",
        tuple((trace_id, name, numeric, missing) for name, numeric, missing in values),
    )


def _insert_event(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    payload: str = '{"secret":"PRIVACY_SENTINEL_service"}',
) -> None:
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
        "VALUES (?,?,?,'STAGED',1,1)",
        (edge_event_id, NOW, payload),
    )


def _insert_incident(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    edge_event_id: str,
    decision_trace_id: str | None,
    manifest_sha: str | None = None,
) -> None:
    if decision_trace_id is None:
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, lifecycle_state, created_at, updated_at
            ) VALUES (?,?,'camera-a','fall',?,'NOT_RECORDED','STAGING',?,?)
            """,
            (incident_id, edge_event_id, NOW, NOW, NOW),
        )
        return
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            runtime_manifest_sha256, decision_trace_id, module_qualified_id,
            policy_qualified_id, effective_policy_id, provenance_state,
            lifecycle_state, created_at, updated_at
        ) VALUES (?,?,'camera-a','fall',?,?,?,'fall.v1','fall.policy.v1',?,
                  'QUALIFIED','STAGING',?,?)
        """,
        (
            incident_id,
            edge_event_id,
            NOW,
            manifest_sha,
            decision_trace_id,
            POLICY_ID,
            NOW,
            NOW,
        ),
    )


def _insert_ref(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    decision_trace_id: str,
) -> None:
    connection.execute(
        "INSERT INTO evidence_event_trace_refs VALUES (?,?)",
        (edge_event_id, decision_trace_id),
    )


def _insert_current_review(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    clip_id: str,
    disposition: str,
    actor_id: str = ACTOR_SENTINEL,
    notes: str | None = NOTES_SENTINEL,
    history_notes: str | None = HISTORY_SENTINEL,
) -> None:
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, state_version) "
        "VALUES (?, 'VERIFIED', 1)",
        (clip_id,),
    )
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, 'NOT_RECORDED', '[]', 'MISSING', ?)
        """,
        (incident_id, clip_id, NOW),
    )
    connection.execute(
        """
        INSERT INTO control_evidence_review_revisions (
            review_id, incident_id, clip_id, review_version, actor_id,
            reviewed_at, disposition, notes
        ) VALUES (?, ?, ?, 1, ?, ?, 'TRUE_POSITIVE', ?)
        """,
        (f"review:{incident_id}:1", incident_id, clip_id, actor_id, NOW, history_notes),
    )
    connection.execute(
        "INSERT INTO control_evidence_review_state "
        "(incident_id, clip_id, current_version) VALUES (?, ?, 1)",
        (incident_id, clip_id),
    )
    connection.execute(
        """
        INSERT INTO control_evidence_review_revisions (
            review_id, incident_id, clip_id, review_version, actor_id,
            reviewed_at, disposition, notes
        ) VALUES (?, ?, ?, 2, ?, ?, ?, ?)
        """,
        (
            f"review:{incident_id}:2",
            incident_id,
            clip_id,
            actor_id,
            NOW,
            disposition,
            notes,
        ),
    )
    connection.execute(
        "UPDATE control_evidence_review_state SET current_version = 2 WHERE incident_id = ?",
        (incident_id,),
    )


def _present(value: object) -> dict[str, object]:
    return {"value": value, "missing_reason": None}


def _missing(reason: str) -> dict[str, object]:
    return {"value": None, "missing_reason": reason}


def _schema_complete_payload() -> dict[str, object]:
    return {
        "decision_provenance": "COMPLETE",
        "decision_provenance_reasons": [],
        "edge_event_id": "event:schema",
        "facility_id": _missing("facility_id_not_a_first_class_column"),
        "camera_id": CAMERA_ID,
        "domain": "fall",
        "event_type": "fall",
        "detected_at": NOW,
        "worker_boot_id": _present(BOOT_ID),
        "stream_epoch": _present(1),
        "frame_seq": _present(40),
        "decision_trace_id": _present(TRACE_A),
        "reason": _present("fall-onset"),
        "previous_state": _present("clear"),
        "current_state": _present("fall"),
        "triggered": _present(True),
        "probability": _present(0.91),
        "threshold": _present(0.5),
        "decision_values": [
            {"name": "fall_probability", "value": 0.91},
            {"name": "operating_threshold", "value": 0.5},
        ],
        "missing_values": [
            {"name": "containment_ratio", "missing_reason": "domain_inapplicable"},
        ],
        "track_id": _present(7),
        "bed_id": _missing("domain_inapplicable"),
        "config_version": _present(3),
        "policy_qualified_id": _present("fall.policy.v1"),
        "model": _present("c" * 64),
        "detector_version": _present("detector-v1"),
        "runtime_manifest_sha256": _present("d" * 64),
        "worker_build_revision": _present("a" * 40),
        "image_revision": _present("b" * 40),
        "delivery": {
            "status": "COMPLETE",
            "reasons": [],
            "outbox_state": _present("PENDING"),
            "attempt_count": _present(1),
            "last_delivery_disposition": _missing("disposition_not_persisted"),
            "last_http_status": _missing("last_http_status_not_persisted"),
            "backend_event_id": _missing("backend_event_id_not_persisted"),
        },
        "media": {
            "status": "UNAVAILABLE",
            "reasons": ["snapshot_not_recorded", "clip_not_recorded"],
            "snapshot": {"state": "NOT_RECORDED", "missing_reason": "snapshot_not_recorded"},
            "clip": {"state": "NOT_RECORDED", "missing_reason": "clip_not_recorded"},
        },
        "review": {
            "status": "UNAVAILABLE",
            "reasons": ["review_not_recorded"],
            "disposition": _missing("review_not_recorded"),
        },
        "neighborhood": {
            "status": "PARTIAL",
            "reasons": ["sequence_gap"],
            "neighborhood_pruned": True,
            "retained_frame_count": 1,
        },
        "correlation": {
            "status": "UNAVAILABLE",
            "reasons": ["alert_correlation_export_not_supplied"],
            "alert_id": _missing("alert_correlation_export_not_supplied"),
        },
    }


def test_foundation_store_projects_one_identity_decision_runtime_set(tmp_path: Path) -> None:
    # Given one migrated event joined to one incident, decision, values, and runtime
    database = _migrated(tmp_path)
    canonical_json, manifest_sha = _current_manifest()
    connection = _connect(database)
    try:
        _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        _insert_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=12)
        _insert_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=ANALYSIS_A,
            manifest_sha=manifest_sha,
        )
        _insert_event(connection, edge_event_id="event:store")
        _insert_incident(
            connection,
            incident_id="incident:store",
            edge_event_id="event:store",
            decision_trace_id=TRACE_A,
            manifest_sha=manifest_sha,
        )
        _insert_ref(connection, edge_event_id="event:store", decision_trace_id=TRACE_A)
        connection.commit()
    finally:
        connection.close()

    # When the committed store seam resolves that edge_event_id
    result = EventExplanationQuery(database).get("event:store")

    # Then it returns exactly one identity/decision/runtime fact set
    assert isinstance(result, EventExplanationFacts)
    assert result.identity.edge_event_id == "event:store"
    assert result.identity.camera_id == CAMERA_ID
    assert result.identity.event_type == "fall"
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_A
    assert result.decision.reason == "fall-onset"
    assert result.decision.track_id == 7
    assert result.decision.bed_missing_reason == "not-applicable"
    assert result.runtime is not None
    assert result.runtime.worker_boot_id == BOOT_ID
    assert result.runtime.frame_seq == 12


def test_foundation_store_unknown_event_is_none(tmp_path: Path) -> None:
    # Given a migrated database with no matching event
    database = _migrated(tmp_path)

    # When the store looks up an unknown id
    result = EventExplanationQuery(database).get("event:missing")

    # Then the seam returns None rather than UNAVAILABLE facts
    assert result is None


def test_foundation_store_unequal_refs_are_typed_conflict(tmp_path: Path) -> None:
    # Given incident and ref rows that store unequal decision_trace_id values
    database = _migrated(tmp_path)
    canonical_json, manifest_sha = _current_manifest()
    connection = _connect(database)
    try:
        _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        _insert_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=1)
        other_analysis = hashlib.sha256(b"analysis-b").hexdigest()
        _insert_analysis(connection, analysis_id=other_analysis, frame_seq=2)
        _insert_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=ANALYSIS_A,
            manifest_sha=manifest_sha,
        )
        _insert_decision(
            connection,
            trace_id=TRACE_B,
            analysis_id=other_analysis,
            manifest_sha=manifest_sha,
        )
        _insert_event(connection, edge_event_id="event:conflict")
        _insert_incident(
            connection,
            incident_id="incident:conflict",
            edge_event_id="event:conflict",
            decision_trace_id=TRACE_A,
            manifest_sha=manifest_sha,
        )
        _insert_ref(connection, edge_event_id="event:conflict", decision_trace_id=TRACE_B)
        connection.commit()
    finally:
        connection.close()

    # When the store resolves the event
    result = EventExplanationQuery(database).get("event:conflict")

    # Then unequal refs are a typed TRACE_REF_CONFLICT
    assert isinstance(result, TraceRefConflict)
    assert result.code == "TRACE_REF_CONFLICT"
    assert result.incident_decision_trace_id == TRACE_A
    assert result.ref_decision_trace_id == TRACE_B


def test_foundation_manifest_projects_current_schema_scalars() -> None:
    # Given a current schema-1 runtime manifest identity
    canonical_json, manifest_sha = _current_manifest()

    # When the committed projector reads that identity
    projection = project_runtime_manifest(
        canonical_json=canonical_json,
        runtime_manifest_sha256=manifest_sha,
        module_qualified_id="fall.v1",
    )

    # Then only allowlisted scalars are present
    assert projection.config_version == 3
    assert projection.policy_version == "fall.policy.v1"
    assert projection.model_version == "c" * 64
    assert projection.detector_version == "detector-v1"
    assert projection.runtime_manifest_sha256 == manifest_sha
    assert projection.worker_build_revision == "a" * 40
    assert projection.image_revision == "b" * 40
    assert "canonical_json" not in projection.as_dict()


def test_foundation_sections_pending_delivery_has_no_backend_event(tmp_path: Path) -> None:
    # Given a staged event with no delivery attempt and no alert export
    database = _migrated(tmp_path)
    canonical_json, manifest_sha = _current_manifest()
    connection = _connect(database)
    try:
        _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        _insert_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=8)
        _insert_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=ANALYSIS_A,
            manifest_sha=manifest_sha,
        )
        _insert_event(connection, edge_event_id="event:pending")
        _insert_incident(
            connection,
            incident_id="incident:pending",
            edge_event_id="event:pending",
            decision_trace_id=TRACE_A,
            manifest_sha=manifest_sha,
        )
        connection.commit()
    finally:
        connection.close()

    # When the committed section projector reads that event
    sections = project_explanation_sections(database, "event:pending")

    # Then pending delivery and absent media/alert stay nested and explicit
    assert sections.delivery.status == "COMPLETE"
    assert sections.delivery.backend_event_id.missing_reason == "backend_event_id_not_persisted"
    assert sections.delivery.last_http_status.missing_reason == "last_http_status_not_persisted"
    assert sections.media.snapshot.missing_reason == "snapshot_not_recorded"
    assert sections.media.clip.missing_reason == "clip_not_recorded"
    assert sections.correlation.status == "UNAVAILABLE"
    assert sections.correlation.alert_id.missing_reason == (
        "alert_correlation_export_not_supplied"
    )


def test_foundation_neighborhood_complete_window_is_exactly_30(tmp_path: Path) -> None:
    # Given the trigger plus 29 preceding same-identity analysis rows
    database = _migrated(tmp_path)
    canonical_json, manifest_sha = _current_manifest()
    connection = _connect(database)
    try:
        _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        trigger_analysis = None
        for seq in range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1):
            analysis_id = hashlib.sha256(f"analysis:{seq}".encode()).hexdigest()
            _insert_analysis(connection, analysis_id=analysis_id, frame_seq=seq)
            if seq == TRIGGER_SEQ:
                trigger_analysis = analysis_id
        assert trigger_analysis is not None
        _insert_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=trigger_analysis,
            manifest_sha=manifest_sha,
        )
        connection.commit()
    finally:
        connection.close()

    # When neighborhood coverage is queried from the decision
    coverage = EventNeighborhoodQuery(database).coverage_for_decision(TRACE_A)

    # Then the window is complete and still carries no category
    assert coverage.status == "COMPLETE"
    assert coverage.neighborhood_pruned is False
    assert coverage.retained_frames == EXPECTED_NEIGHBORHOOD_FRAMES
    assert coverage.category is None


def test_foundation_schema_complete_allows_nested_delivery_and_media_gaps() -> None:
    # Given a COMPLETE decision-provenance payload with nested delivery/media gaps
    payload = _schema_complete_payload()

    # When the committed response contract validates it
    parsed = EventExplanationResponse.model_validate(payload)
    dumped = parsed.model_dump(mode="json")

    # Then top-level completeness stays COMPLETE while nested gaps remain typed
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["delivery"]["backend_event_id"]["missing_reason"] == (
        "backend_event_id_not_persisted"
    )
    assert dumped["media"]["status"] == "UNAVAILABLE"
    assert dumped["correlation"]["status"] == "UNAVAILABLE"
    assert dumped["review"]["status"] == "UNAVAILABLE"
    assert dumped["neighborhood"]["neighborhood_pruned"] is True
    assert dumped["bed_id"]["missing_reason"] == "domain_inapplicable"


def _legacy_manifest() -> tuple[str, str]:
    return _canonical({"configuration": {"config_version": 2}})


def _analysis_id_for_seq(seq: int) -> str:
    return hashlib.sha256(f"analysis:{seq}".encode()).hexdigest()


def _seed_event_graph(
    tmp_path: Path,
    *,
    edge_event_id: str,
    neighborhood: bool = False,
    include_trace: bool = True,
    conflict: bool = False,
    legacy_manifest: bool = False,
    drop_analysis: bool = False,
    pending_attempt: bool = False,
) -> Path:
    database = _migrated(tmp_path / edge_event_id.replace(":", "_"))
    canonical_json, manifest_sha = (
        _legacy_manifest() if legacy_manifest else _current_manifest()
    )
    connection = _connect(database)
    try:
        _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        payload = json.dumps(
            {
                "secret": PRIVACY_SENTINEL,
                "path": PATH_SENTINEL,
                "polygon": GEOMETRY_SENTINEL,
            },
            separators=(",", ":"),
        )
        _insert_event(connection, edge_event_id=edge_event_id, payload=payload)
        if not include_trace:
            _insert_incident(
                connection,
                incident_id=f"incident:{edge_event_id}",
                edge_event_id=edge_event_id,
                decision_trace_id=None,
            )
            connection.commit()
            return database
        trigger_analysis = ANALYSIS_A
        if neighborhood:
            for seq in range(TRIGGER_SEQ - PRECEDING_FRAMES, TRIGGER_SEQ + 1):
                analysis_id = _analysis_id_for_seq(seq)
                _insert_analysis(connection, analysis_id=analysis_id, frame_seq=seq)
                if seq == TRIGGER_SEQ:
                    trigger_analysis = analysis_id
        else:
            _insert_analysis(connection, analysis_id=trigger_analysis, frame_seq=TRIGGER_SEQ)
        _insert_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=trigger_analysis,
            manifest_sha=manifest_sha,
        )
        if conflict:
            other_analysis = hashlib.sha256(b"analysis-b").hexdigest()
            _insert_analysis(connection, analysis_id=other_analysis, frame_seq=TRIGGER_SEQ + 1)
            _insert_decision(
                connection,
                trace_id=TRACE_B,
                analysis_id=other_analysis,
                manifest_sha=manifest_sha,
            )
        _insert_incident(
            connection,
            incident_id=f"incident:{edge_event_id}",
            edge_event_id=edge_event_id,
            decision_trace_id=TRACE_A,
            manifest_sha=manifest_sha,
        )
        if drop_analysis:
            connection.execute(
                "DELETE FROM runtime_analysis_traces WHERE trace_id = ?",
                (trigger_analysis,),
            )
        _insert_ref(
            connection,
            edge_event_id=edge_event_id,
            decision_trace_id=TRACE_B if conflict else TRACE_A,
        )
        if pending_attempt:
            connection.execute(
                "UPDATE evidence_events SET state = 'READY', delivery_state = 'PENDING', "
                "attempt_count = 1 WHERE edge_event_id = ?",
                (edge_event_id,),
            )
        connection.commit()
    finally:
        connection.close()
    return database


def _explain(database: Path, edge_event_id: str) -> EventExplanationResponse:
    return EventExplanationService(database).explain(edge_event_id)


def _dump(response: EventExplanationResponse) -> dict[str, object]:
    return response.model_dump(mode="json")


def _serialized(response: EventExplanationResponse) -> str:
    return json.dumps(_dump(response), separators=(",", ":"))


def test_service_complete_pending_delivery_keeps_decision_provenance(
    tmp_path: Path,
) -> None:
    # Given a uniquely resolved current decision/analysis/manifest and pending delivery
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:complete",
        neighborhood=True,
        pending_attempt=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:complete"))

    # Then decision provenance is COMPLETE and pending delivery stays nested
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["decision_provenance_reasons"] == []
    assert dumped["edge_event_id"] == "event:complete"
    assert dumped["camera_id"] == CAMERA_ID
    assert dumped["domain"] == "fall"
    assert dumped["event_type"] == "fall"
    assert dumped["decision_trace_id"]["value"] == TRACE_A
    assert dumped["reason"]["value"] == "fall-onset"
    assert dumped["probability"]["value"] == 0.91
    assert dumped["threshold"]["value"] == 0.5
    assert dumped["facility_id"]["missing_reason"] == "facility_id_not_a_first_class_column"
    assert dumped["delivery"]["status"] == "COMPLETE"
    assert dumped["delivery"]["attempt_count"]["value"] == 1
    assert dumped["delivery"]["backend_event_id"]["missing_reason"] == (
        "backend_event_id_not_persisted"
    )
    assert dumped["delivery"]["last_http_status"]["missing_reason"] == (
        "last_http_status_not_persisted"
    )
    assert dumped["neighborhood"]["status"] == "COMPLETE"
    assert dumped["neighborhood"]["retained_frame_count"] == 30


def test_service_partial_when_required_analysis_group_absent(tmp_path: Path) -> None:
    # Given a uniquely resolved decision whose linked analysis row was retained away
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:partial-analysis",
        drop_analysis=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:partial-analysis"))

    # Then provenance is PARTIAL because the required analysis group is absent
    assert dumped["decision_provenance"] == "PARTIAL"
    assert dumped["decision_provenance_reasons"] == ["analysis_trace_unresolved"]
    assert dumped["decision_trace_id"]["value"] == TRACE_A
    assert dumped["worker_boot_id"]["missing_reason"] == "analysis_trace_unresolved"
    assert dumped["stream_epoch"]["missing_reason"] == "analysis_trace_unresolved"
    assert dumped["frame_seq"]["missing_reason"] == "analysis_trace_unresolved"


def test_service_partial_when_required_manifest_group_is_legacy(tmp_path: Path) -> None:
    # Given a uniquely resolved decision linked to a legacy runtime manifest
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:partial-legacy",
        legacy_manifest=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:partial-legacy"))

    # Then provenance is PARTIAL because the required manifest group is legacy
    assert dumped["decision_provenance"] == "PARTIAL"
    assert dumped["decision_provenance_reasons"] == ["runtime_manifest_unresolved"]
    assert dumped["decision_trace_id"]["value"] == TRACE_A
    assert dumped["config_version"]["missing_reason"] == "legacy_manifest_field"
    assert dumped["detector_version"]["missing_reason"] == "legacy_manifest_field"


def test_service_unavailable_when_event_has_no_decision_trace(tmp_path: Path) -> None:
    # Given an existing event whose incident stores no uniquely resolved trace
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:unavailable",
        include_trace=False,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:unavailable"))

    # Then provenance is UNAVAILABLE with a typed unresolved-trace reason
    assert dumped["decision_provenance"] == "UNAVAILABLE"
    assert dumped["decision_provenance_reasons"] == ["decision_trace_unresolved"]
    assert dumped["decision_trace_id"]["value"] is None
    assert dumped["decision_trace_id"]["missing_reason"] == "decision_trace_unresolved"
    assert dumped["reason"]["missing_reason"] == "decision_trace_unresolved"
    assert dumped["probability"]["value"] is None
    assert dumped["probability"]["missing_reason"] is not None
    assert dumped["edge_event_id"] == "event:unavailable"
    assert dumped["camera_id"] == CAMERA_ID


def test_service_unavailable_on_trace_ref_conflict(tmp_path: Path) -> None:
    # Given incident and ref rows that store unequal decision_trace_id values
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:conflict",
        conflict=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:conflict"))

    # Then provenance is UNAVAILABLE with TRACE_REF_CONFLICT and no chosen trace
    assert dumped["decision_provenance"] == "UNAVAILABLE"
    assert dumped["decision_provenance_reasons"] == ["trace_ref_conflict"]
    assert dumped["decision_trace_id"]["value"] is None
    assert dumped["decision_trace_id"]["missing_reason"] == "trace_ref_conflict"
    assert dumped["reason"]["missing_reason"] == "trace_ref_conflict"
    assert dumped["worker_boot_id"]["missing_reason"] == "trace_ref_conflict"


def test_service_unknown_event_is_typed_not_found(tmp_path: Path) -> None:
    # Given a migrated database with no matching evidence_events row
    database = _migrated(tmp_path)

    # When the service looks up an unknown id
    # Then the outcome is typed not-found, not UNAVAILABLE
    with pytest.raises(EventExplanationNotFound) as error:
        _explain(database, "event:missing")
    assert error.value.edge_event_id == "event:missing"


def test_service_absent_alert_and_snapshot_do_not_downgrade_complete(
    tmp_path: Path,
) -> None:
    # Given a complete decision explanation with no snapshot, clip, or alert export
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:absent-media",
        neighborhood=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:absent-media"))

    # Then nested media/alert gaps stay nested and do not change COMPLETE
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["media"]["status"] != "COMPLETE"
    assert dumped["media"]["snapshot"]["missing_reason"] == "snapshot_not_recorded"
    assert dumped["media"]["clip"]["missing_reason"] == "clip_not_recorded"
    assert dumped["correlation"]["status"] == "UNAVAILABLE"
    assert dumped["correlation"]["alert_id"]["missing_reason"] == (
        "alert_correlation_export_not_supplied"
    )
    assert dumped["review"]["status"] == "UNAVAILABLE"
    assert dumped["review"]["disposition"]["missing_reason"] == "review_not_recorded"


def test_service_current_review_is_independent_of_decision_completeness(
    tmp_path: Path,
) -> None:
    # Given a reviewed event whose decision trace is incomplete
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:reviewed-partial",
        drop_analysis=True,
    )
    connection = _connect(database)
    try:
        _insert_current_review(
            connection,
            incident_id="incident:event:reviewed-partial",
            clip_id="clip:reviewed-partial",
            disposition="FALSE_POSITIVE",
        )
        persisted = connection.execute(
            """
            SELECT review.disposition, review_state.current_version
            FROM control_evidence_review_state AS review_state
            JOIN control_evidence_review_revisions AS review
              ON review.incident_id = review_state.incident_id
             AND review.clip_id = review_state.clip_id
             AND review.review_version = review_state.current_version
            WHERE review_state.incident_id = ?
            """,
            ("incident:event:reviewed-partial",),
        ).fetchone()
        assert persisted == ("FALSE_POSITIVE", 2)
        connection.commit()
    finally:
        connection.close()

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:reviewed-partial"))
    rendered = json.dumps(dumped, separators=(",", ":"))

    # Then review truth stays COMPLETE while decision provenance stays PARTIAL
    assert dumped["decision_provenance"] == "PARTIAL"
    assert dumped["decision_provenance_reasons"] == ["analysis_trace_unresolved"]
    assert dumped["review"]["status"] == "COMPLETE"
    assert dumped["review"]["reasons"] == []
    assert dumped["review"]["disposition"]["value"] == "FALSE_POSITIVE"
    assert dumped["review"]["disposition"]["missing_reason"] is None
    assert ACTOR_SENTINEL not in rendered
    assert NOTES_SENTINEL not in rendered
    assert HISTORY_SENTINEL not in rendered
    assert "actor_id" not in rendered
    assert '"notes"' not in rendered


def test_service_unreviewed_event_keeps_typed_review_not_recorded(
    tmp_path: Path,
) -> None:
    # Given a uniquely resolved event with no current review state
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:unreviewed",
        neighborhood=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:unreviewed"))

    # Then the review section remains typed UNAVAILABLE with review_not_recorded
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["review"]["status"] == "UNAVAILABLE"
    assert dumped["review"]["reasons"] == ["review_not_recorded"]
    assert dumped["review"]["disposition"]["value"] is None
    assert dumped["review"]["disposition"]["missing_reason"] == "review_not_recorded"


def test_service_malformed_persisted_review_disposition_fails_closed() -> None:
    from backend.app.features.evidence.explanation_service import project_current_review
    from backend.app.features.evidence.explanation_store import EventExplanationCurrentReview

    # Given a current-review projection whose disposition is not an approved token.
    # Schema v16 CHECK already forbids storing that text; the composer must still
    # fail closed if an unapproved value is ever projected.
    review = EventExplanationCurrentReview(
        disposition="NOT_A_DISPOSITION",
        current_version=2,
    )

    # When that projection is mapped onto the public review section
    section = project_current_review(review)
    dumped = section.model_dump(mode="json")
    rendered = json.dumps(dumped, separators=(",", ":"))

    # Then the raw disposition is never echoed and the section fails closed
    assert dumped["status"] == "UNAVAILABLE"
    assert dumped["reasons"] == ["persisted_value_invalid"]
    assert dumped["disposition"]["value"] is None
    assert dumped["disposition"]["missing_reason"] == "persisted_value_invalid"
    assert "NOT_A_DISPOSITION" not in rendered


def test_service_domain_inapplicable_values_keep_complete(tmp_path: Path) -> None:
    # Given a complete fall explanation whose bed facts are domain-inapplicable
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:inapplicable",
        neighborhood=True,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:inapplicable"))

    # Then typed inapplicable nulls remain in-section and provenance stays COMPLETE
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["bed_id"]["value"] is None
    assert dumped["bed_id"]["missing_reason"] == "domain_inapplicable"
    assert any(
        item["name"] == "containment_ratio"
        and item["missing_reason"] == "domain_inapplicable"
        for item in dumped["missing_values"]
    )
    assert dumped["track_id"]["value"] == 7


def test_service_incomplete_neighborhood_stays_nested(tmp_path: Path) -> None:
    # Given a complete decision whose 30-frame neighborhood is only the trigger
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:pruned-neighborhood",
        neighborhood=False,
    )

    # When the service composes an explanation
    dumped = _dump(_explain(database, "event:pruned-neighborhood"))

    # Then neighborhood incompleteness is nested and does not change COMPLETE
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["neighborhood"]["status"] != "COMPLETE"
    assert dumped["neighborhood"]["neighborhood_pruned"] is True
    assert dumped["neighborhood"]["retained_frame_count"] == 1
    assert dumped["neighborhood"]["reasons"]


def test_service_rejects_malformed_edge_event_id(tmp_path: Path) -> None:
    # Given a migrated database
    database = _migrated(tmp_path)
    service = EventExplanationService(database)

    # When the caller supplies malformed identities
    # Then the service rejects them before composing a response
    with pytest.raises(ValueError, match="edge_event_id"):
        service.explain("")
    with pytest.raises(ValueError, match="edge_event_id"):
        service.explain("event\x00hidden")
    with pytest.raises(ValueError, match="edge_event_id"):
        service.explain("e" * 257)


def test_service_never_leaks_raw_payload_path_or_geometry(tmp_path: Path) -> None:
    # Given a complete event whose payload contains unique forbidden sentinels
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:private",
        neighborhood=True,
    )

    # When the service serializes the composed response
    response = _explain(database, "event:private")
    rendered = _serialized(response) + repr(response)

    # Then raw payload, path, and geometry sentinels never appear
    assert PRIVACY_SENTINEL not in rendered
    assert PATH_SENTINEL not in rendered
    assert GEOMETRY_SENTINEL not in rendered
    assert "payload_json" not in rendered
    assert "canonical_json" not in rendered


def _selected_rtsp_sentinel() -> str:
    return "".join(
        (
            "rtsp",
            "://",
            "user",
            ":",
            "SERVICE_pass_9e44",
            "@",
            "10.255.255.2",
            "/stream",
        )
    )


def test_service_malformed_selected_policy_id_is_typed_unavailable(
    tmp_path: Path,
) -> None:
    # Given a complete event whose selected policy_qualified_id is untrusted RTSP text
    secret = _selected_rtsp_sentinel()
    database = _seed_event_graph(
        tmp_path,
        edge_event_id="event:poisoned-policy",
        neighborhood=True,
    )
    connection = _connect(database)
    try:
        updated = connection.execute(
            "UPDATE evidence_decision_traces SET policy_qualified_id = ?",
            (secret,),
        ).rowcount
        assert updated == 1
        connection.commit()
    finally:
        connection.close()

    # When the service composes an explanation
    response = _explain(database, "event:poisoned-policy")
    dumped = _dump(response)
    rendered = _serialized(response) + repr(response)

    # Then the identifier is typed unavailable and the exact secret is never echoed
    assert dumped["policy_qualified_id"]["value"] is None
    assert dumped["policy_qualified_id"]["missing_reason"] == "persisted_value_invalid"
    assert dumped["decision_provenance"] == "COMPLETE"
    assert secret not in rendered
    leaked = {"policy_qualified_id": {"value": secret}}
    with pytest.raises(AssertionError):
        assert leaked["policy_qualified_id"]["value"] is None
        assert dumped["policy_qualified_id"]["value"] is None


def test_service_hostile_selected_policy_ids_are_typed_unavailable(
    tmp_path: Path,
) -> None:
    # Given complete events whose selected policy IDs are residual hostile classes
    secrets = (
        "/private/service-policy-path.bin",
        "ghp_" + ("B" * 36),
        "z" * 257,
        "fall.policy\x07.v1",
        "f\u0430ll.policy.v1",
    )
    for index, secret in enumerate(secrets):
        edge_event_id = f"event:hostile-policy-{index}"
        database = _seed_event_graph(
            tmp_path / edge_event_id,
            edge_event_id=edge_event_id,
            neighborhood=True,
        )
        connection = _connect(database)
        try:
            updated = connection.execute(
                "UPDATE evidence_decision_traces SET policy_qualified_id = ?",
                (secret,),
            ).rowcount
            assert updated == 1
            stored = connection.execute(
                "SELECT policy_qualified_id FROM evidence_decision_traces"
            ).fetchone()
            assert stored is not None
            assert stored[0] == secret
            connection.commit()
        finally:
            connection.close()

        response = _explain(database, edge_event_id)
        dumped = _dump(response)
        rendered = _serialized(response) + repr(response)

        assert dumped["policy_qualified_id"]["value"] is None
        assert dumped["policy_qualified_id"]["missing_reason"] == (
            "persisted_value_invalid"
        )
        assert dumped["decision_provenance"] == "COMPLETE"
        assert secret not in rendered
