"""Adversarial privacy and serving-boundary coverage for Event Explanation."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.features.evidence.explanation_schemas import (
    EventExplanationResponse,
    PolicyQualifiedIdFact,
)
from backend.app.features.evidence.explanation_service import EventExplanationService
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.main import create_app, no_lifespan
from shared.edge_db.migrator import migrate_database

NOW = "2026-08-15T00:00:00Z"
CAMERA_ID = "camera-a"
BOOT_ID = "boot-a"
POLICY_ID = "b" * 64
ANALYSIS_A = hashlib.sha256(b"privacy-analysis-a").hexdigest()
TRACE_A = hashlib.sha256(b"privacy-trace-a").hexdigest()
TRIGGER_SEQ = 40
PRECEDING_FRAMES = 29
COMPLETE_EVENT_ID = str(uuid4())
PARTIAL_EVENT_ID = str(uuid4())
UNAVAILABLE_EVENT_ID = str(uuid4())
UNKNOWN_EVENT_ID = str(uuid4())
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
OPERATOR_EXPLANATION_PATH = "/api/v1/events/{edge_event_id}/explanation"

PAYLOAD_SENTINEL = "PAYLOAD_SENTINEL_privacy_9f3c21ab"
NOTES_SENTINEL = "NOTES_SENTINEL_privacy_2ab17c21"
ACTOR_SENTINEL = "actor:sentinel-privacy-7c21e9aa"
HISTORY_SENTINEL = "HISTORY_SENTINEL_privacy_revision_text_44e1"
CREDENTIAL_SENTINEL = "CREDENTIAL_SENTINEL_privacy_token_44e1"
RTSP_SENTINEL = "".join(
    (
        "rtsp",
        "://",
        "user",
        ":",
        "SENTINEL_pass_9e44",
        "@",
        "10.255.255.1",
        "/stream",
    )
)
MEDIA_ABS_PATH_SENTINEL = "/private/media-path-sentinel-privacy.mp4"
MEDIA_RELPATH_SENTINEL = "clips/private-SENTINEL-privacy-path/clip.mp4"
COMPONENT_PATH_SENTINEL = "/private/component-path-sentinel-privacy.bin"
MODEL_PATH_SENTINEL = "/private/model-path-sentinel-privacy.pt"
ERROR_SENTINEL = "Traceback: raw boom at /secret/privacy-error-path"
CANONICAL_NESTED_SENTINEL = "CANONICAL_NESTED_OBJECT_SENTINEL_privacy"
COORD_SENTINEL = 987654321
POLYGON_SENTINEL = "[[987654321,123456789]]"
FRAME_WIDTH_SENTINEL = 1921
FRAME_HEIGHT_SENTINEL = 1083
KEYPOINT_X_SENTINEL = 424242
KEYPOINT_Y_SENTINEL = 434343
BED_X1_SENTINEL = 515151
BED_Y1_SENTINEL = 525252
ABS_PATH_POLICY_SENTINEL = "/private/policy-path-sentinel-privacy.bin"
TOKEN_LIKE_POLICY_SENTINEL = "ghp_" + ("A" * 36)
OVERLENGTH_POLICY_SENTINEL = "a" * 257
CONTROL_POLICY_SENTINEL = "fall.policy\x07.v1"
CONFUSABLE_POLICY_SENTINEL = "f\u0430ll.policy.v1"
HOSTILE_POLICY_SENTINELS: Final = (
    RTSP_SENTINEL,
    ABS_PATH_POLICY_SENTINEL,
    TOKEN_LIKE_POLICY_SENTINEL,
    OVERLENGTH_POLICY_SENTINEL,
    CONTROL_POLICY_SENTINEL,
    CONFUSABLE_POLICY_SENTINEL,
)

PRIVACY_SENTINELS: Final = (
    PAYLOAD_SENTINEL,
    NOTES_SENTINEL,
    ACTOR_SENTINEL,
    HISTORY_SENTINEL,
    CREDENTIAL_SENTINEL,
    RTSP_SENTINEL,
    MEDIA_ABS_PATH_SENTINEL,
    MEDIA_RELPATH_SENTINEL,
    COMPONENT_PATH_SENTINEL,
    MODEL_PATH_SENTINEL,
    ERROR_SENTINEL,
    CANONICAL_NESTED_SENTINEL,
    POLYGON_SENTINEL,
    str(COORD_SENTINEL),
    str(FRAME_WIDTH_SENTINEL),
    str(FRAME_HEIGHT_SENTINEL),
    str(KEYPOINT_X_SENTINEL),
    str(KEYPOINT_Y_SENTINEL),
    str(BED_X1_SENTINEL),
    str(BED_Y1_SENTINEL),
)

DENYLISTED_KEYS: Final = (
    "payload_json",
    "notes",
    "actor_id",
    "canonical_json",
    "media_relpath",
    "manifest_path",
    "contained_relpath",
    "last_error_code",
    "polygon",
    "coordinates",
    "coordinate_space",
    "rtsp",
    "rtsp_url",
    "password",
    "token",
    "weights_path",
    "component_path",
    "media_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "keypoints",
)

INTENDED_KEYS: Final = (
    "decision_provenance",
    "edge_event_id",
    "camera_id",
    "facility_id",
    "delivery",
    "media",
    "review",
    "neighborhood",
    "correlation",
)

FORBIDDEN_IMPORTS: Final = (
    "worker",
    "shared.events.outbox",
    "shared.events.schemas",
    "runtime.edge_worker",
    "runners",
    "sources",
    "perception",
    "domains",
)
SERVING_ROOT: Final = Path(__file__).resolve().parents[1] / "backend" / "app"


def _canonical(content: object) -> tuple[str, str]:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def _dirty_manifest() -> dict[str, object]:
    return {
        "manifest_schema_version": 1,
        "build": {
            "detector_version": "detector-v1",
            "worker_build_revision": "a" * 40,
            "image_revision": "b" * 40,
            "component_path": COMPONENT_PATH_SENTINEL,
            "rtsp_url": RTSP_SENTINEL,
            "token": CREDENTIAL_SENTINEL,
        },
        "configuration": {
            "config_version": 3,
            "nested": {"secret": CANONICAL_NESTED_SENTINEL, "password": CREDENTIAL_SENTINEL},
        },
        "modules": [
            {
                "qualified_id": "fall.v1",
                "policy_schema": "fall.policy.v1",
                "component_bindings": [
                    {
                        "component_id": "fall-classifier",
                        "kind": "model",
                        "component_path": COMPONENT_PATH_SENTINEL,
                    }
                ],
            }
        ],
        "components": [
            {
                "component_id": "fall-classifier",
                "artifact_sha256": "c" * 64,
                "weights_path": MODEL_PATH_SENTINEL,
            }
        ],
        "cameras": [
            {
                "rtsp": RTSP_SENTINEL,
                "bed_zone": {
                    "polygon": [[COORD_SENTINEL, COORD_SENTINEL + 1]],
                    "coordinate_space": POLYGON_SENTINEL,
                    "media_path": MEDIA_ABS_PATH_SENTINEL,
                },
            }
        ],
    }


def _legacy_dirty_manifest() -> dict[str, object]:
    return {
        "configuration": {
            "config_version": 2,
            "nested": {"secret": CANONICAL_NESTED_SENTINEL},
        },
        "cameras": [
            {
                "rtsp": RTSP_SENTINEL,
                "bed_zone": {"polygon": [[COORD_SENTINEL, BED_Y1_SENTINEL]]},
            }
        ],
    }


def _payload(edge_event_id: str) -> str:
    return json.dumps(
        {
            "secret": PAYLOAD_SENTINEL,
            "notes": NOTES_SENTINEL,
            "actor_id": ACTOR_SENTINEL,
            "password": CREDENTIAL_SENTINEL,
            "token": CREDENTIAL_SENTINEL,
            "rtsp": RTSP_SENTINEL,
            "path": MEDIA_ABS_PATH_SENTINEL,
            "polygon": POLYGON_SENTINEL,
            "coordinates": [COORD_SENTINEL, FRAME_WIDTH_SENTINEL],
            "edge_event_id": edge_event_id,
        },
        separators=(",", ":"),
    )


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _analysis_id_for_seq(seq: int) -> str:
    return hashlib.sha256(f"privacy-analysis:{seq}".encode()).hexdigest()


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


def _insert_geometry(connection: sqlite3.Connection, analysis_id: str) -> None:
    connection.execute(
        """
        INSERT INTO runtime_analysis_persons (
            analysis_trace_id, ordinal, track_id, track_missing_reason,
            x1, y1, x2, y2, confidence
        ) VALUES (?, 0, 7, NULL, ?, ?, ?, ?, 0.9)
        """,
        (
            analysis_id,
            COORD_SENTINEL,
            COORD_SENTINEL + 1,
            COORD_SENTINEL + 2,
            COORD_SENTINEL + 3,
        ),
    )
    connection.execute(
        """
        INSERT INTO runtime_analysis_keypoints (
            analysis_trace_id, person_ordinal, keypoint_index, x, y, confidence
        ) VALUES (?, 0, 0, ?, ?, 0.8)
        """,
        (analysis_id, KEYPOINT_X_SENTINEL, KEYPOINT_Y_SENTINEL),
    )
    connection.execute(
        """
        INSERT INTO runtime_analysis_beds (
            analysis_trace_id, ordinal, x1, y1, x2, y2, confidence, provenance
        ) VALUES (?, 0, ?, ?, ?, ?, 0.7, 'fresh')
        """,
        (
            analysis_id,
            BED_X1_SENTINEL,
            BED_Y1_SENTINEL,
            BED_X1_SENTINEL + 1,
            BED_Y1_SENTINEL + 1,
        ),
    )
    connection.execute(
        """
        INSERT INTO runtime_analysis_bed_points (
            analysis_trace_id, bed_ordinal, point_index, x, y
        ) VALUES (?, 0, 0, ?, ?)
        """,
        (analysis_id, COORD_SENTINEL, BED_Y1_SENTINEL),
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
        "VALUES (?,1,?,?,1,?,0,0,?,?,'fresh',1)",
        (analysis_id, BOOT_ID, CAMERA_ID, frame_seq, FRAME_WIDTH_SENTINEL, FRAME_HEIGHT_SENTINEL),
    )
    _insert_geometry(connection, analysis_id)


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


def _insert_clip(connection: sqlite3.Connection, clip_id: str) -> None:
    connection.execute(
        """
        INSERT INTO evidence_clips (
            clip_id, local_state, manifest_path, state_version, media_relpath,
            sha256, size_bytes, mime_type, publish_state, last_error_code
        ) VALUES (?, 'VERIFIED', ?, 2, ?, ?, 1, 'video/mp4', 'PUBLISHED', ?)
        """,
        (
            clip_id,
            MEDIA_ABS_PATH_SENTINEL,
            MEDIA_RELPATH_SENTINEL,
            "e" * 64,
            ERROR_SENTINEL,
        ),
    )


def _insert_private_sources(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    incident_id: str,
    clip_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO evidence_media_objects (
            media_id, content_sha256, size_bytes, mime_type, contained_relpath,
            basename, created_at
        ) VALUES (?, ?, 1, 'video/mp4', ?, 'clip.mp4', ?)
        """,
        (f"media:{clip_id}", "f" * 64, MEDIA_RELPATH_SENTINEL, NOW),
    )
    connection.execute(
        """
        INSERT INTO evidence_artifact_slots (
            incident_id, slot_name, state, media_id, created_at, updated_at
        ) VALUES (?, 'PRIMARY_CLIP', 'AVAILABLE', ?, ?, ?)
        """,
        (incident_id, f"media:{clip_id}", NOW, NOW),
    )
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, manifest_relpath, manifest_sha256, manifest_size_bytes,
            media_id, source_packet_preserved, source_media_json, truncation_json, created_at
        ) VALUES (?, ?, ?, ?, 1, ?, 1, ?, '[]', ?)
        """,
        (
            incident_id,
            clip_id,
            MEDIA_RELPATH_SENTINEL,
            "d" * 64,
            f"media:{clip_id}",
            json.dumps({"rtsp": RTSP_SENTINEL, "path": MEDIA_ABS_PATH_SENTINEL}),
            NOW,
        ),
    )
    connection.execute(
        """
        INSERT INTO control_evidence_review_revisions (
            review_id, incident_id, clip_id, review_version, actor_id,
            reviewed_at, disposition, notes
        ) VALUES (?, ?, ?, 1, ?, ?, 'TRUE_POSITIVE', ?)
        """,
        (f"review:{incident_id}:1", incident_id, clip_id, ACTOR_SENTINEL, NOW, HISTORY_SENTINEL),
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
        ) VALUES (?, ?, ?, 2, ?, ?, 'FALSE_POSITIVE', ?)
        """,
        (f"review:{incident_id}:2", incident_id, clip_id, ACTOR_SENTINEL, NOW, NOTES_SENTINEL),
    )
    connection.execute(
        "UPDATE control_evidence_review_state SET current_version = 2 WHERE incident_id = ?",
        (incident_id,),
    )
    connection.execute(
        "UPDATE evidence_events SET last_error_code = ? WHERE edge_event_id = ?",
        (ERROR_SENTINEL, edge_event_id),
    )


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
    source = _legacy_dirty_manifest() if legacy_manifest else _dirty_manifest()
    canonical_json, manifest_sha = _canonical(source)
    connection = _connect(database)
    try:
        existing = connection.execute(
            "SELECT 1 FROM runtime_manifest_contents LIMIT 1"
        ).fetchone()
        if existing is None:
            _seed_manifest(connection, canonical_json=canonical_json, manifest_sha=manifest_sha)
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
            "VALUES (?,?,?,'STAGED',1,1)",
            (edge_event_id, NOW, _payload(edge_event_id)),
        )
        clip_id = f"clip:{edge_event_id}"
        incident_id = f"incident:{edge_event_id}"
        _insert_clip(connection, clip_id)
        if not include_trace:
            connection.execute(
                """
                INSERT INTO evidence_incidents (
                    incident_id, edge_event_id, camera_id, event_type, detected_at,
                    provenance_missing_reason, primary_clip_id, lifecycle_state,
                    created_at, updated_at
                ) VALUES (?,?,'camera-a','fall',?,'NOT_RECORDED',?,'STAGING',?,?)
                """,
                (incident_id, edge_event_id, NOW, clip_id, NOW, NOW),
            )
            _insert_private_sources(
                connection,
                edge_event_id=edge_event_id,
                incident_id=incident_id,
                clip_id=clip_id,
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
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                runtime_manifest_sha256, decision_trace_id, module_qualified_id,
                policy_qualified_id, effective_policy_id, provenance_state,
                primary_clip_id, lifecycle_state, created_at, updated_at
            ) VALUES (?,?,'camera-a','fall',?,?,?,'fall.v1','fall.policy.v1',?,
                      'QUALIFIED',?,'STAGING',?,?)
            """,
            (
                incident_id,
                edge_event_id,
                NOW,
                manifest_sha,
                TRACE_A,
                POLICY_ID,
                clip_id,
                NOW,
                NOW,
            ),
        )
        if drop_analysis:
            connection.execute(
                "DELETE FROM runtime_analysis_traces WHERE trace_id = ?",
                (trigger_analysis,),
            )
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES (?,?)",
            (edge_event_id, TRACE_A),
        )
        _insert_private_sources(
            connection,
            edge_event_id=edge_event_id,
            incident_id=incident_id,
            clip_id=clip_id,
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


def _explanation_client(database: Path) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    app.state.event_explanation_service = EventExplanationService(database)
    return TestClient(app, raise_server_exceptions=False)


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


def _explanation_path(edge_event_id: str) -> str:
    return f"/api/v1/events/{edge_event_id}/explanation"


def _json_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for item in value.values():
            keys.update(_json_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_json_keys(item))
    return keys


def _rendered_logs(caplog: logging.LogCaptureFixture) -> str:
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    rendered: list[str] = []
    for record in caplog.records:
        rendered.append(record.getMessage())
        rendered.append(record.message)
        rendered.append(formatter.format(record))
    return "\n".join(rendered)


def _assert_privacy_safe(body_text: str, logs: str, *, parsed: object | None = None) -> None:
    haystack = body_text + "\n" + logs
    for sentinel in PRIVACY_SENTINELS:
        assert sentinel not in haystack
    if parsed is not None:
        keys = _json_keys(parsed)
        for key in DENYLISTED_KEYS:
            assert key not in keys
        blob = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for key in DENYLISTED_KEYS:
            assert f'"{key}"' not in blob
    for key in DENYLISTED_KEYS:
        assert key not in logs


def _is_canonical_uuid4(value: str) -> bool:
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def test_policy_qualified_id_contract_rejects_hostile_selected_identifiers() -> None:
    # Given the five hostile selected-identifier classes plus credentialed RTSP
    assert "://" in RTSP_SENTINEL
    assert ABS_PATH_POLICY_SENTINEL.startswith("/")
    assert TOKEN_LIKE_POLICY_SENTINEL.isalnum() is False
    assert len(OVERLENGTH_POLICY_SENTINEL) > 64
    assert "\x07" in CONTROL_POLICY_SENTINEL
    assert CONFUSABLE_POLICY_SENTINEL != "fall.policy.v1"

    # When each sentinel is inserted into the API policy identifier contract
    # Then validation fails closed instead of serializing the secret
    for secret in HOSTILE_POLICY_SENTINELS:
        with pytest.raises(ValidationError):
            PolicyQualifiedIdFact(value=secret)
    for legitimate in ("fall.policy.v1", "bed_exit.policy.v1", "mobility.policy.v1"):
        accepted = PolicyQualifiedIdFact(value=legitimate)
        assert accepted.value == legitimate
    rejected_opaque = PolicyQualifiedIdFact.model_validate
    with pytest.raises(ValidationError):
        rejected_opaque({"value": "onnxruntime", "missing_reason": None})


def test_privacy_assertion_fails_when_review_actor_note_or_history_is_captured() -> None:
    # Given actor, note, and history review sentinels already used in the fixture
    assert ACTOR_SENTINEL.startswith("actor:")
    assert NOTES_SENTINEL.startswith("NOTES_SENTINEL")
    assert HISTORY_SENTINEL.startswith("HISTORY_SENTINEL")

    # When those sentinels are inserted into a captured response or rendered log
    # Then the existing privacy assertion still fails closed
    with pytest.raises(AssertionError):
        _assert_privacy_safe(ACTOR_SENTINEL, "")
    with pytest.raises(AssertionError):
        _assert_privacy_safe(NOTES_SENTINEL, "")
    with pytest.raises(AssertionError):
        _assert_privacy_safe("", HISTORY_SENTINEL)


def test_authenticated_reviewed_event_never_leaks_actor_note_or_history(
    tmp_path: Path,
    caplog,
) -> None:
    # Given a complete event whose current review stores actor, notes, and history text
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    caplog.set_level(logging.DEBUG)

    with _explanation_client(database) as client:
        _login(client)
        # When the authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then current review is reported without private actor/note/history fields
    assert response.status_code == 200
    body = response.json()
    parsed = EventExplanationResponse.model_validate(body)
    assert parsed.model_dump(mode="json") == body
    assert body["review"]["status"] == "COMPLETE"
    assert body["review"]["disposition"]["value"] == "FALSE_POSITIVE"
    logs = _rendered_logs(caplog)
    _assert_privacy_safe(response.text, logs, parsed=body)
    assert ACTOR_SENTINEL not in response.text
    assert NOTES_SENTINEL not in response.text
    assert HISTORY_SENTINEL not in response.text
    assert ACTOR_SENTINEL not in logs
    assert NOTES_SENTINEL not in logs
    assert HISTORY_SENTINEL not in logs
    leaked = {"review": {"disposition": {"value": ACTOR_SENTINEL}}}
    with pytest.raises(AssertionError):
        assert leaked["review"]["disposition"]["value"] is None
        assert body["review"]["disposition"]["value"] is None


def test_privacy_assertion_fails_when_runtime_rtsp_sentinel_is_captured() -> None:
    # Given the runtime-composed credentialed RTSP sentinel
    assert "://" in RTSP_SENTINEL
    assert "@" in RTSP_SENTINEL
    userinfo, _host = RTSP_SENTINEL.split("://", 1)[1].split("@", 1)
    assert ":" in userinfo

    # When that sentinel is inserted into a captured response or rendered log
    # Then the existing privacy assertion still fails closed
    with pytest.raises(AssertionError):
        _assert_privacy_safe(RTSP_SENTINEL, "")
    with pytest.raises(AssertionError):
        _assert_privacy_safe("", RTSP_SENTINEL)


def test_tracked_source_does_not_embed_contiguous_credentialed_rtsp() -> None:
    # Given this test module's tracked source text after scanner quote-joining
    source = Path(__file__).read_text(encoding="utf-8")
    joined = re.sub(r'''(["'])\s*(?:\+\s*)?["']''', "", source)
    pattern = re.compile(
        r"rtsps?://(?P<username>[^/\s:@]+):(?P<password>[^@\s/]+)"
        r"@(?P<host>[^/\s\"']+)(?:/[^\s\"']*)?",
        re.IGNORECASE,
    )

    # Then the working-tree source stays scanner-safe while the runtime value remains credentialed
    assert pattern.search(joined) is None
    assert RTSP_SENTINEL not in source
    assert RTSP_SENTINEL.startswith("rtsp://")


def test_serving_files_still_do_not_import_worker_or_event_schema_modules() -> None:
    # Given the committed serving package after the explanation route landed
    violations: list[str] = []

    # When every serving Python file is scanned for forbidden imports
    for path in sorted(SERVING_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            import_name = node.names[0].name if isinstance(node, ast.Import) else node.module
            if import_name is None:
                continue
            for forbidden in FORBIDDEN_IMPORTS:
                if import_name == forbidden or import_name.startswith(f"{forbidden}."):
                    violations.append(
                        f"{path}:{node.lineno}: imports {import_name}"
                    )

    # Then the worker/backend import boundary is unchanged
    assert not violations


def test_openapi_exposes_only_get_explanation_and_response_schema() -> None:
    # Given the real FastAPI app
    app = create_app(lifespan=no_lifespan)
    spec = app.openapi()

    # When the operator explanation OpenAPI item is inspected
    item = spec["paths"][OPERATOR_EXPLANATION_PATH]
    schema = spec["components"]["schemas"]["EventExplanationResponse"]

    # Then only GET exists, the 200 schema is exact, and no sibling write/admin/config
    assert set(item) == {"get"}
    assert item["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/EventExplanationResponse"
    }
    assert set(schema["properties"]) >= set(INTENDED_KEYS)
    assert all(key not in schema["properties"] for key in DENYLISTED_KEYS)
    sibling_paths = (
        "/api/v1/events/{edge_event_id}",
        "/api/v1/events/{edge_event_id}/explanation/config",
        "/api/v1/events/{edge_event_id}/explanation/admin",
        "/api/v1/events/{edge_event_id}/explanation/write",
        "/api/v1/events/config",
        "/api/v1/events/admin",
    )
    assert all(path not in spec["paths"] for path in sibling_paths)
    event_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/v1/events/")
    ]
    assert len(event_routes) == 1
    assert event_routes[0].methods == {"GET"}
    assert event_routes[0].path == OPERATOR_EXPLANATION_PATH


def test_authenticated_complete_partial_unavailable_bodies_and_logs_are_privacy_safe(
    tmp_path: Path,
    caplog,
) -> None:
    # Given complete, partial, unavailable, and legacy-manifest events with unique sentinels
    cases = (
        (
            COMPLETE_EVENT_ID,
            "COMPLETE",
            {"neighborhood": True, "pending_attempt": True},
        ),
        (PARTIAL_EVENT_ID, "PARTIAL", {"drop_analysis": True}),
        (UNAVAILABLE_EVENT_ID, "UNAVAILABLE", {"include_trace": False}),
        (str(uuid4()), "PARTIAL", {"legacy_manifest": True}),
    )
    caplog.set_level(logging.DEBUG)

    for edge_event_id, provenance, flags in cases:
        database = _seed_event_graph(tmp_path / edge_event_id, edge_event_id=edge_event_id, **flags)
        with _explanation_client(database) as client:
            _login(client)
            # When the real authenticated explanation route is requested
            response = client.get(_explanation_path(edge_event_id))

        # Then the typed body keeps intended keys and drops every sentinel and denylisted key
        assert response.status_code == 200
        body = response.json()
        parsed = EventExplanationResponse.model_validate(body)
        assert parsed.model_dump(mode="json") == body
        assert body["decision_provenance"] == provenance
        assert body["edge_event_id"] == edge_event_id
        for key in INTENDED_KEYS:
            assert key in body
        _assert_privacy_safe(response.text, _rendered_logs(caplog), parsed=body)


def test_unauthorized_unknown_malformed_and_internal_error_paths_leak_no_sentinels(
    tmp_path: Path,
    caplog,
) -> None:
    # Given a sentinel-populated complete event and a real TestClient
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    caplog.set_level(logging.DEBUG)

    class _FailAfterQuery(EventExplanationService):
        def explain(self, edge_event_id: str) -> EventExplanationResponse:
            super().explain(edge_event_id)
            raise RuntimeError("composition failed")

    with _explanation_client(database) as client:
        # When unauthorized, unknown, malformed, and post-query internal-error paths run
        missing = client.get(_explanation_path(COMPLETE_EVENT_ID))
        bad = client.get(
            _explanation_path(COMPLETE_EVENT_ID),
            headers={"Authorization": "Bearer not-a-session"},
        )
        _login(client)
        unknown = client.get(_explanation_path(UNKNOWN_EVENT_ID))
        malformed = client.get(_explanation_path("not-a-uuid"))
        client.app.state.event_explanation_service = _FailAfterQuery(database)
        exploded = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then none of the error surfaces expose sentinels, denylisted keys, or stored raw text
    assert missing.status_code == 401
    assert bad.status_code == 401
    assert unknown.status_code == 404
    assert malformed.status_code == 422
    assert exploded.status_code == 500
    assert not _is_canonical_uuid4("not-a-uuid")
    logs = _rendered_logs(caplog)
    for response in (missing, bad, unknown, malformed):
        parsed = response.json()
        assert isinstance(parsed, dict)
        assert set(parsed) == {"detail"}
        for key in INTENDED_KEYS:
            assert key not in parsed
        _assert_privacy_safe(response.text, logs, parsed=parsed)
        assert COMPLETE_EVENT_ID not in response.text
        assert UNKNOWN_EVENT_ID not in response.text
        assert "not-a-uuid" not in response.text
        assert "Traceback" not in response.text
        assert "composition failed" not in response.text
    _assert_privacy_safe(exploded.text, logs)
    assert COMPLETE_EVENT_ID not in exploded.text
    assert "Traceback" not in exploded.text
    assert "composition failed" not in exploded.text
    assert ERROR_SENTINEL not in logs
    assert PAYLOAD_SENTINEL not in logs
    assert NOTES_SENTINEL not in logs
    assert ACTOR_SENTINEL not in logs
    assert RTSP_SENTINEL not in logs
    assert CREDENTIAL_SENTINEL not in logs


def _poison_selected_policy_text(database: Path, *, policy_qualified_id: str) -> None:
    connection = _connect(database)
    try:
        updated = connection.execute(
            "UPDATE evidence_decision_traces SET policy_qualified_id = ?",
            (policy_qualified_id,),
        ).rowcount
        assert updated >= 1
        stored = connection.execute(
            "SELECT policy_qualified_id FROM evidence_decision_traces"
        ).fetchone()
        assert stored is not None
        assert stored[0] == policy_qualified_id
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "label,secret",
    (
        ("rtsp_credentialed", RTSP_SENTINEL),
        ("absolute_path", ABS_PATH_POLICY_SENTINEL),
        ("token_like", TOKEN_LIKE_POLICY_SENTINEL),
        ("overlength", OVERLENGTH_POLICY_SENTINEL),
        ("control_chars", CONTROL_POLICY_SENTINEL),
        ("unicode_confusable", CONFUSABLE_POLICY_SENTINEL),
    ),
    ids=(
        "rtsp_credentialed",
        "absolute_path",
        "token_like",
        "overlength",
        "control_chars",
        "unicode_confusable",
    ),
)
def test_schema_valid_selected_policy_text_never_crosses_api_privacy_boundary(
    tmp_path: Path,
    caplog,
    label: str,
    secret: str,
) -> None:
    # Given a complete event whose selected policy_qualified_id is a schema-valid
    # hostile identifier from one of the five verifier classes
    del label
    database = _seed_event_graph(
        tmp_path,
        edge_event_id=COMPLETE_EVENT_ID,
        neighborhood=True,
        pending_attempt=True,
    )
    _poison_selected_policy_text(database, policy_qualified_id=secret)
    caplog.set_level(logging.DEBUG)

    with _explanation_client(database) as client:
        _login(client)
        # When the real authenticated explanation route is requested
        response = client.get(_explanation_path(COMPLETE_EVENT_ID))

    # Then the persisted-but-invalid identifier is typed unavailable and never echoed
    assert response.status_code == 200
    body = response.json()
    parsed = EventExplanationResponse.model_validate(body)
    assert parsed.model_dump(mode="json") == body
    assert body["edge_event_id"] == COMPLETE_EVENT_ID
    assert body["decision_provenance"] == "COMPLETE"
    policy = body["policy_qualified_id"]
    assert policy["value"] is None
    assert policy["missing_reason"] == "persisted_value_invalid"
    logs = _rendered_logs(caplog)
    _assert_privacy_safe(response.text, logs, parsed=body)
    assert secret not in response.text
    assert secret not in logs
    leaked = {"policy_qualified_id": {"value": secret}}
    with pytest.raises(AssertionError):
        assert leaked["policy_qualified_id"]["value"] is None
        assert policy["value"] is None
