"""Dashboard-authenticated Event Explanation HTTP surface."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.features.evidence.explanation_schemas import EventExplanationResponse
from backend.app.features.evidence.explanation_service import EventExplanationService
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.main import create_app, no_lifespan
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
PRIVACY_SENTINEL = "PRIVACY_SENTINEL_api_8f21c9de"
NOTES_SENTINEL = "OPERATOR_NOTES_api_8f21c9de"
ACTOR_SENTINEL = "actor:sentinel-api-7c21e9aa"
HISTORY_SENTINEL = "HISTORY_SENTINEL_api_revision_text_44e1"
PATH_SENTINEL = "/private/media-path-sentinel.mp4"
GEOMETRY_SENTINEL = "[[987654,123456]]"
COMPLETE_EVENT_ID = str(uuid4())
PARTIAL_EVENT_ID = str(uuid4())
UNAVAILABLE_EVENT_ID = str(uuid4())
UNKNOWN_EVENT_ID = str(uuid4())
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
FORBIDDEN_TOKENS = (
    "payload_json",
    PRIVACY_SENTINEL,
    NOTES_SENTINEL,
    ACTOR_SENTINEL,
    HISTORY_SENTINEL,
    PATH_SENTINEL,
    GEOMETRY_SENTINEL,
    "canonical_json",
    "actor_id",
)
COMPLETENESS_LEAK_TOKENS = (
    "decision_provenance",
    "COMPLETE",
    "PARTIAL",
    "UNAVAILABLE",
    "decision_trace_unresolved",
    "analysis_trace_unresolved",
    "runtime_manifest_unresolved",
)


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


def _legacy_manifest() -> tuple[str, str]:
    return _canonical({"configuration": {"config_version": 2}})


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
        (
            (trace_id, "fall_probability", 0.91, None),
            (trace_id, "operating_threshold", 0.5, None),
            (trace_id, "containment_ratio", None, "not-applicable"),
        ),
    )


def _insert_event(connection: sqlite3.Connection, *, edge_event_id: str) -> None:
    payload = json.dumps(
        {
            "secret": PRIVACY_SENTINEL,
            "notes": NOTES_SENTINEL,
            "path": PATH_SENTINEL,
            "polygon": GEOMETRY_SENTINEL,
        },
        separators=(",", ":"),
    )
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


def _analysis_id_for_seq(seq: int) -> str:
    return hashlib.sha256(f"analysis:{seq}".encode()).hexdigest()


def _seed_event_graph(
    tmp_path: Path,
    *,
    edge_event_id: str,
    neighborhood: bool = False,
    include_trace: bool = True,
    drop_analysis: bool = False,
    pending_attempt: bool = False,
    legacy_manifest: bool = False,
) -> Path:
    database = tmp_path / "edge.sqlite3"
    if not database.exists():
        migrate_database(database)
    canonical_json, manifest_sha = (
        _legacy_manifest() if legacy_manifest else _current_manifest()
    )
    connection = _connect(database)
    try:
        existing = connection.execute(
            "SELECT 1 FROM runtime_manifest_contents LIMIT 1"
        ).fetchone()
        if existing is None:
            _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        _insert_event(connection, edge_event_id=edge_event_id)
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
                already = connection.execute(
                    "SELECT 1 FROM runtime_analysis_traces WHERE trace_id = ?",
                    (analysis_id,),
                ).fetchone()
                if already is None:
                    _insert_analysis(connection, analysis_id=analysis_id, frame_seq=seq)
                if seq == TRIGGER_SEQ:
                    trigger_analysis = analysis_id
        else:
            already = connection.execute(
                "SELECT 1 FROM runtime_analysis_traces WHERE trace_id = ?",
                (trigger_analysis,),
            ).fetchone()
            if already is None:
                _insert_analysis(
                    connection, analysis_id=trigger_analysis, frame_seq=TRIGGER_SEQ
                )
        decision_exists = connection.execute(
            "SELECT 1 FROM evidence_decision_traces WHERE trace_id = ?",
            (TRACE_A,),
        ).fetchone()
        if decision_exists is None:
            _insert_decision(
                connection,
                trace_id=TRACE_A,
                analysis_id=trigger_analysis,
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
        _insert_ref(connection, edge_event_id=edge_event_id, decision_trace_id=TRACE_A)
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


def _explanation_client(
    database: Path,
    *,
    explanation_service: EventExplanationService | None = None,
) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    if explanation_service is not None:
        app.state.event_explanation_service = explanation_service
    return TestClient(app, raise_server_exceptions=False)


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


def _explanation_path(edge_event_id: str) -> str:
    return f"/api/v1/events/{edge_event_id}/explanation"


def _is_canonical_uuid4(value: str) -> bool:
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _assert_no_privacy_or_completeness_leak(text: str) -> None:
    for token in (*FORBIDDEN_TOKENS, *COMPLETENESS_LEAK_TOKENS):
        assert token not in text


def test_operator_incident_route_requires_dashboard_session_and_returns_existing(
    tmp_path: Path, caplog
) -> None:
    # Given a migrated operator incident and a real FastAPI app
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        include_trace=False,
    )
    del database
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        # When the existing operator evidence route is requested without a session
        unauthorized = client.get(f"/api/v1/incidents/{COMPLETE_EVENT_ID}")
        # Then the current dashboard-auth seam rejects it before any incident body
        assert unauthorized.status_code == 401
        _assert_no_privacy_or_completeness_leak(unauthorized.text + caplog.text)

        # When a valid dashboard session is established
        _login(client)
        authorized = client.get(f"/api/v1/incidents/{COMPLETE_EVENT_ID}")

    # Then the existing operator route still returns the seeded incident
    assert authorized.status_code == 200
    body = authorized.json()
    assert body["edge_event_id"] == COMPLETE_EVENT_ID
    assert body["incident_id"] == f"incident:{COMPLETE_EVENT_ID}"
    assert "payload_json" not in authorized.text


def test_authenticated_complete_explanation_returns_200_with_pending_delivery(
    tmp_path: Path,
) -> None:
    # Given a uniquely resolved current decision with pending delivery
    _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        # When the dashboard-authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then the committed service response is returned exactly as 200
    assert response.status_code == 200
    body = response.json()
    parsed = EventExplanationResponse.model_validate(body)
    assert parsed.model_dump(mode="json") == body
    assert body["decision_provenance"] == "COMPLETE"
    assert body["decision_provenance_reasons"] == []
    assert body["edge_event_id"] == COMPLETE_EVENT_ID
    assert body["camera_id"] == CAMERA_ID
    assert body["delivery"]["status"] == "COMPLETE"
    assert body["delivery"]["attempt_count"]["value"] == 1
    assert body["delivery"]["backend_event_id"]["missing_reason"] == (
        "backend_event_id_not_persisted"
    )
    assert body["neighborhood"]["status"] == "COMPLETE"
    assert "payload_json" not in response.text
    assert PRIVACY_SENTINEL not in response.text
    assert body["review"]["status"] == "UNAVAILABLE"
    assert body["review"]["disposition"]["missing_reason"] == "review_not_recorded"


def test_authenticated_reviewed_event_reports_current_disposition(
    tmp_path: Path, caplog
) -> None:
    # Given a schema-v16 current FALSE_POSITIVE review on an existing event
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    connection = _connect(database)
    try:
        _insert_current_review(
            connection,
            incident_id=f"incident:{COMPLETE_EVENT_ID}",
            clip_id=f"clip:{COMPLETE_EVENT_ID}",
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
            (f"incident:{COMPLETE_EVENT_ID}",),
        ).fetchone()
        assert persisted == ("FALSE_POSITIVE", 2)
        connection.commit()
    finally:
        connection.close()
    caplog.set_level(logging.DEBUG)

    with _explanation_client(database) as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then the current review is reported truthfully without private fields
    assert response.status_code == 200
    body = EventExplanationResponse.model_validate(response.json()).model_dump(mode="json")
    assert body["decision_provenance"] == "COMPLETE"
    assert body["review"]["status"] == "COMPLETE"
    assert body["review"]["reasons"] == []
    assert body["review"]["disposition"]["value"] == "FALSE_POSITIVE"
    assert body["review"]["disposition"]["missing_reason"] is None
    haystack = response.text + caplog.text
    for token in (PRIVACY_SENTINEL, NOTES_SENTINEL, ACTOR_SENTINEL, HISTORY_SENTINEL):
        assert token not in haystack
    assert "payload_json" not in response.text
    assert "actor_id" not in response.text
    assert NOTES_SENTINEL not in response.text
    assert ACTOR_SENTINEL not in response.text
    assert HISTORY_SENTINEL not in response.text
    leaked = {"review": {"disposition": {"value": NOTES_SENTINEL}}}
    with pytest.raises(AssertionError):
        assert leaked["review"]["disposition"]["value"] is None
        assert body["review"]["disposition"]["value"] is None


def test_authenticated_reviewed_partial_event_keeps_review_truth(
    tmp_path: Path,
) -> None:
    # Given a reviewed event whose required analysis row is absent
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=PARTIAL_EVENT_ID,
        drop_analysis=True,
    )
    connection = _connect(database)
    try:
        _insert_current_review(
            connection,
            incident_id=f"incident:{PARTIAL_EVENT_ID}",
            clip_id=f"clip:{PARTIAL_EVENT_ID}",
            disposition="TRUE_POSITIVE",
        )
        connection.commit()
    finally:
        connection.close()

    with _explanation_client(database) as client:
        _login(client)
        response = client.get(_explanation_path(PARTIAL_EVENT_ID))

    # Then review availability stays independent from decision completeness
    assert response.status_code == 200
    body = EventExplanationResponse.model_validate(response.json()).model_dump(mode="json")
    assert body["decision_provenance"] == "PARTIAL"
    assert body["review"]["status"] == "COMPLETE"
    assert body["review"]["disposition"]["value"] == "TRUE_POSITIVE"
    assert NOTES_SENTINEL not in response.text
    assert ACTOR_SENTINEL not in response.text


def test_authenticated_unreviewed_event_reports_review_not_recorded(
    tmp_path: Path,
) -> None:
    # Given an existing event with no current review state
    _seed_event_graph(
        tmp_path,
        edge_event_id=UNAVAILABLE_EVENT_ID,
        include_trace=False,
    )

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        response = client.get(_explanation_path(UNAVAILABLE_EVENT_ID))

    # Then the unreviewed event remains typed UNAVAILABLE with review_not_recorded
    assert response.status_code == 200
    body = EventExplanationResponse.model_validate(response.json()).model_dump(mode="json")
    assert body["decision_provenance"] == "UNAVAILABLE"
    assert body["review"]["status"] == "UNAVAILABLE"
    assert body["review"]["reasons"] == ["review_not_recorded"]
    assert body["review"]["disposition"]["missing_reason"] == "review_not_recorded"


def test_authenticated_partial_explanation_returns_200(tmp_path: Path) -> None:
    # Given a uniquely resolved decision whose linked analysis row is absent
    _seed_event_graph(
        tmp_path,
        edge_event_id=PARTIAL_EVENT_ID,
        drop_analysis=True,
    )

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(PARTIAL_EVENT_ID))

    # Then PARTIAL provenance is returned as 200 without hiding the gap
    assert response.status_code == 200
    body = EventExplanationResponse.model_validate(response.json()).model_dump(mode="json")
    assert body["decision_provenance"] == "PARTIAL"
    assert body["decision_provenance_reasons"] == ["analysis_trace_unresolved"]
    assert body["decision_trace_id"]["value"] == TRACE_A
    assert body["worker_boot_id"]["missing_reason"] == "analysis_trace_unresolved"


def test_authenticated_unavailable_explanation_returns_200(tmp_path: Path) -> None:
    # Given an existing event with no uniquely resolved decision trace
    _seed_event_graph(
        tmp_path,
        edge_event_id=UNAVAILABLE_EVENT_ID,
        include_trace=False,
    )

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(UNAVAILABLE_EVENT_ID))

    # Then UNAVAILABLE provenance is returned as 200, not 404
    assert response.status_code == 200
    body = EventExplanationResponse.model_validate(response.json()).model_dump(mode="json")
    assert body["decision_provenance"] == "UNAVAILABLE"
    assert body["decision_provenance_reasons"] == ["decision_trace_unresolved"]
    assert body["decision_trace_id"]["missing_reason"] == "decision_trace_unresolved"
    assert body["edge_event_id"] == UNAVAILABLE_EVENT_ID


def test_authenticated_unknown_uuid_returns_404_without_privacy_leak(
    tmp_path: Path, caplog
) -> None:
    # Given a migrated database with no matching event
    _migrated(tmp_path)
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        # When an unknown canonical UUID is requested
        response = client.get(_explanation_path(UNKNOWN_EVENT_ID))

    # Then the typed not-found maps to 404 without leaking completeness or payload
    assert response.status_code == 404
    assert response.json() == {"detail": "event not found"}
    _assert_no_privacy_or_completeness_leak(response.text + caplog.text)
    assert UNKNOWN_EVENT_ID not in response.text
    assert "Traceback" not in response.text


def test_authenticated_malformed_path_id_returns_privacy_safe_422(
    tmp_path: Path, caplog
) -> None:
    # Given a migrated database and a dashboard session
    _migrated(tmp_path)
    malformed_ids = ("not-a-uuid", "event:opaque", "e" * 257)
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        _login(client)
        # When malformed path IDs are requested after auth
        responses = [client.get(_explanation_path(value)) for value in malformed_ids]

    # Then the current public ID contract maps them to privacy-safe 422
    for value, response in zip(malformed_ids, responses, strict=True):
        assert not _is_canonical_uuid4(value)
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid edge_event_id"}
        _assert_no_privacy_or_completeness_leak(response.text + caplog.text)
        assert value not in response.text
        assert "Traceback" not in response.text


def test_missing_or_bad_token_returns_401_before_existence_or_completeness(
    tmp_path: Path, caplog
) -> None:
    # Given a complete existing event
    _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        # When the route is requested without a session or with a forged bearer
        missing = client.get(_explanation_path(COMPLETE_EVENT_ID))
        bad = client.get(
            _explanation_path(COMPLETE_EVENT_ID),
            headers={"Authorization": "Bearer not-a-session"},
        )

    # Then both fail closed at 401 before leaking completeness or existence
    assert missing.status_code == 401
    assert bad.status_code == 401
    _assert_no_privacy_or_completeness_leak(missing.text + bad.text + caplog.text)


def test_malformed_path_id_without_auth_still_returns_401(tmp_path: Path, caplog) -> None:
    # Given a migrated database and a malformed path identity
    _migrated(tmp_path)
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        # When a malformed ID is requested without a dashboard session
        response = client.get(_explanation_path("not-a-uuid"))

    # Then auth fails first; validation details are not distinguishable
    assert response.status_code == 401
    _assert_no_privacy_or_completeness_leak(response.text + caplog.text)
    assert "invalid edge_event_id" not in response.text
    assert "not-a-uuid" not in response.text


def test_unknown_id_without_auth_still_returns_401(tmp_path: Path, caplog) -> None:
    # Given a migrated database and an unknown canonical UUID
    _migrated(tmp_path)
    caplog.set_level(logging.DEBUG)

    with _explanation_client(tmp_path / "edge.sqlite3") as client:
        # When an unknown ID is requested without a dashboard session
        response = client.get(_explanation_path(UNKNOWN_EVENT_ID))

    # Then auth fails first; existence is not distinguishable
    assert response.status_code == 401
    _assert_no_privacy_or_completeness_leak(response.text + caplog.text)
    assert UNKNOWN_EVENT_ID not in response.text


class _FailingExplanationService(EventExplanationService):
    def __init__(self, database_path: Path, error: BaseException) -> None:
        super().__init__(database_path)
        self._error = error

    def explain(self, edge_event_id: str) -> EventExplanationResponse:
        del edge_event_id
        raise self._error


def test_service_valueerror_on_valid_uuid_is_generic_500_not_422(
    tmp_path: Path, caplog
) -> None:
    # Given a syntactically valid existing UUID whose composer raises ValueError
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    failure = ValueError("some unexpected composition failure")
    caplog.set_level(logging.DEBUG)

    with _explanation_client(
        database,
        explanation_service=_FailingExplanationService(database, failure),
    ) as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then the app generic 500 path is used; the body is not a path-ID 422
    assert response.status_code == 500
    assert response.status_code != 422
    assert "invalid edge_event_id" not in response.text
    assert "some unexpected composition failure" not in response.text
    _assert_no_privacy_or_completeness_leak(response.text)
    assert COMPLETE_EVENT_ID not in response.text
    assert "Traceback" not in response.text


def test_service_validationerror_on_valid_uuid_is_generic_500_not_422(
    tmp_path: Path, caplog
) -> None:
    # Given a syntactically valid existing UUID whose composer raises ValidationError
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    try:
        EventExplanationResponse.model_validate({})
    except ValidationError as error:
        failure = error
    caplog.set_level(logging.DEBUG)

    with _explanation_client(
        database,
        explanation_service=_FailingExplanationService(database, failure),
    ) as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then the app generic 500 path is used; schema internals are not leaked as 422
    assert response.status_code == 500
    assert response.status_code != 422
    assert "invalid edge_event_id" not in response.text
    assert "Field required" not in response.text
    assert "EventExplanationResponse" not in response.text
    _assert_no_privacy_or_completeness_leak(response.text)
    assert COMPLETE_EVENT_ID not in response.text
    assert "Traceback" not in response.text


def test_openapi_exposes_exact_explanation_path() -> None:
    # Given the real FastAPI app
    app = create_app(lifespan=no_lifespan)

    # When OpenAPI paths are collected
    paths = set(app.openapi()["paths"])

    # Then the explanation route is mounted exactly under /api/v1
    assert "/api/v1/events/{edge_event_id}/explanation" in paths
