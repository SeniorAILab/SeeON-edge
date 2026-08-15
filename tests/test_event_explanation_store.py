from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import pytest

from backend.app.features.evidence.record_store import CentralEvidenceQuery
from shared.edge_db.migrator import migrate_database
from shared.edge_db.schema import SCHEMA_VERSION

NOW = "2026-08-13T00:00:00Z"
PRIVACY_SENTINEL = "PRIVACY_SENTINEL_event_payload_9f3c21ab"
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
ANALYSIS_A = hashlib.sha256(b"analysis-a").hexdigest()
ANALYSIS_B = hashlib.sha256(b"analysis-b").hexdigest()
TRACE_A = hashlib.sha256(b"trace-a").hexdigest()
TRACE_B = hashlib.sha256(b"trace-b").hexdigest()


def _migrated(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _seed_manifest(connection: sqlite3.Connection) -> None:
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


def _seed_analysis(
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
        "VALUES (?,1,'boot-a','camera-a',1,?,0,0,320,180,'fresh',1)",
        (analysis_id, frame_seq),
    )


def _seed_decision(
    connection: sqlite3.Connection,
    *,
    trace_id: str,
    analysis_id: str,
    values: tuple[tuple[str, float | None, str | None], ...] = (
        ("fall_probability", 0.91, None),
        ("operating_threshold", None, "adapter-not-provided"),
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
        (trace_id, analysis_id, POLICY_ID, MANIFEST_ID),
    )
    connection.executemany(
        "INSERT INTO evidence_decision_values "
        "(decision_trace_id, name, numeric_value, missing_reason) VALUES (?,?,?,?)",
        tuple((trace_id, name, numeric, missing) for name, numeric, missing in values),
    )


def _seed_event(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    detected_at: str = NOW,
) -> None:
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
        "VALUES (?,?,?,'STAGED',1,1)",
        (edge_event_id, detected_at, f'{{"secret":"{PRIVACY_SENTINEL}"}}'),
    )


def _seed_incident(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    edge_event_id: str,
    decision_trace_id: str | None,
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
            MANIFEST_ID,
            decision_trace_id,
            POLICY_ID,
            NOW,
            NOW,
        ),
    )


def _seed_ref(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    decision_trace_id: str,
) -> None:
    connection.execute(
        "INSERT INTO evidence_event_trace_refs VALUES (?,?)",
        (edge_event_id, decision_trace_id),
    )


def _query():
    from backend.app.features.evidence.explanation_store import EventExplanationQuery

    return EventExplanationQuery


def test_central_evidence_query_identity_remains_privacy_bounded_for_event_review(
    tmp_path: Path,
) -> None:
    # Given a schema-v16 event+incident whose payload_json holds a privacy sentinel
    assert SCHEMA_VERSION == 16
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_event(connection, edge_event_id="event:review")
        _seed_incident(
            connection,
            incident_id="incident:review",
            edge_event_id="event:review",
            decision_trace_id=None,
        )
        connection.commit()

    # When the existing central evidence query loads that identity
    summary = CentralEvidenceQuery(database).get("event:review")

    # Then exactly one privacy-bounded review projection is returned
    assert summary is not None
    assert summary.edge_event_id == "event:review"
    assert summary.incident_id == "incident:review"
    assert summary.camera_id == "camera-a"
    assert summary.event_type == "fall"
    assert summary.decision_trace_id is None
    assert not hasattr(summary, "payload_json")
    assert not hasattr(summary, "facility_id")
    assert PRIVACY_SENTINEL not in repr(summary)


def test_explanation_query_rejects_malformed_edge_event_identity(tmp_path: Path) -> None:
    # Given a migrated database
    database = _migrated(tmp_path)
    query = _query()(database)

    # When the caller supplies malformed edge_event_id values
    # Then the new seam rejects them before any row projection
    with pytest.raises(ValueError, match="edge_event_id"):
        query.get("")
    with pytest.raises(ValueError, match="edge_event_id"):
        query.get("event\x00hidden")
    with pytest.raises(ValueError, match="edge_event_id"):
        query.get("e" * 257)


def test_explanation_query_projects_one_identity_decision_runtime_fact_set(
    tmp_path: Path,
) -> None:
    from backend.app.features.evidence.explanation_store import EventExplanationFacts

    # Given one event joined to one incident, one shared trace, values, and runtime
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=12)
        _seed_decision(connection, trace_id=TRACE_A, analysis_id=ANALYSIS_A)
        _seed_event(connection, edge_event_id="event:complete")
        _seed_incident(
            connection,
            incident_id="incident:complete",
            edge_event_id="event:complete",
            decision_trace_id=TRACE_A,
        )
        _seed_ref(connection, edge_event_id="event:complete", decision_trace_id=TRACE_A)
        connection.commit()

    # When the explanation query loads that edge_event_id
    result = _query()(database).get("event:complete")

    # Then exactly one immutable identity/decision/runtime fact set is returned
    assert isinstance(result, EventExplanationFacts)
    assert result.identity.edge_event_id == "event:complete"
    assert result.identity.incident_id == "incident:complete"
    assert result.identity.camera_id == "camera-a"
    assert result.identity.event_type == "fall"
    assert result.identity.detected_at == NOW
    assert not hasattr(result.identity, "facility_id")
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_A
    assert result.decision.reason == "fall-onset"
    assert result.decision.previous_state == "clear"
    assert result.decision.current_state == "fall"
    assert result.decision.triggered is True
    assert result.decision.track_id == 7
    assert result.decision.bed_id is None
    assert result.decision.bed_missing_reason == "not-applicable"
    projected_values = tuple(
        (item.name, item.numeric_value, item.missing_reason)
        for item in result.decision.values
    )
    assert projected_values == (
        ("fall_probability", 0.91, None),
        ("operating_threshold", None, "adapter-not-provided"),
    )
    assert result.runtime is not None
    assert result.runtime.analysis_trace_id == ANALYSIS_A
    assert result.runtime.worker_boot_id == "boot-a"
    assert result.runtime.camera_id == "camera-a"
    assert result.runtime.stream_epoch == 1
    assert result.runtime.frame_seq == 12


def test_explanation_query_resolves_incident_only_decision_trace(tmp_path: Path) -> None:
    from backend.app.features.evidence.explanation_store import EventExplanationFacts

    # Given an incident decision_trace_id and no evidence_event_trace_refs row
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=3)
        _seed_decision(connection, trace_id=TRACE_A, analysis_id=ANALYSIS_A)
        _seed_event(connection, edge_event_id="event:incident-only")
        _seed_incident(
            connection,
            incident_id="incident:incident-only",
            edge_event_id="event:incident-only",
            decision_trace_id=TRACE_A,
        )
        connection.commit()

    # When the explanation query resolves the event
    result = _query()(database).get("event:incident-only")

    # Then the singly present incident trace is used
    assert isinstance(result, EventExplanationFacts)
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_A
    assert result.runtime is not None
    assert result.runtime.analysis_trace_id == ANALYSIS_A


def test_explanation_query_resolves_ref_only_decision_trace(tmp_path: Path) -> None:
    from backend.app.features.evidence.explanation_store import EventExplanationFacts

    # Given a trace ref and an incident that stores no decision_trace_id
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_B, frame_seq=4)
        _seed_decision(connection, trace_id=TRACE_B, analysis_id=ANALYSIS_B)
        _seed_event(connection, edge_event_id="event:ref-only")
        _seed_incident(
            connection,
            incident_id="incident:ref-only",
            edge_event_id="event:ref-only",
            decision_trace_id=None,
        )
        _seed_ref(connection, edge_event_id="event:ref-only", decision_trace_id=TRACE_B)
        connection.commit()

    # When the explanation query resolves the event
    result = _query()(database).get("event:ref-only")

    # Then the singly present ref trace is used
    assert isinstance(result, EventExplanationFacts)
    assert result.identity.incident_id == "incident:ref-only"
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_B
    assert result.runtime is not None
    assert result.runtime.frame_seq == 4


def test_explanation_query_accepts_equal_incident_and_ref_trace_ids(tmp_path: Path) -> None:
    from backend.app.features.evidence.explanation_store import EventExplanationFacts

    # Given incident and ref rows that store the same decision_trace_id
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=8)
        _seed_decision(connection, trace_id=TRACE_A, analysis_id=ANALYSIS_A)
        _seed_event(connection, edge_event_id="event:equal-refs")
        _seed_incident(
            connection,
            incident_id="incident:equal-refs",
            edge_event_id="event:equal-refs",
            decision_trace_id=TRACE_A,
        )
        _seed_ref(connection, edge_event_id="event:equal-refs", decision_trace_id=TRACE_A)
        connection.commit()

    # When the explanation query resolves the event
    result = _query()(database).get("event:equal-refs")

    # Then the equal IDs collapse to one decision fact set
    assert isinstance(result, EventExplanationFacts)
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_A


def test_explanation_query_returns_typed_trace_ref_conflict_for_unequal_ids(
    tmp_path: Path,
) -> None:
    from backend.app.features.evidence.explanation_store import (
        EventExplanationFacts,
        TraceRefConflict,
    )

    # Given incident and ref rows that store unequal decision_trace_id values
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=1)
        _seed_analysis(connection, analysis_id=ANALYSIS_B, frame_seq=2)
        _seed_decision(connection, trace_id=TRACE_A, analysis_id=ANALYSIS_A)
        _seed_decision(connection, trace_id=TRACE_B, analysis_id=ANALYSIS_B)
        _seed_event(connection, edge_event_id="event:conflict")
        _seed_incident(
            connection,
            incident_id="incident:conflict",
            edge_event_id="event:conflict",
            decision_trace_id=TRACE_A,
        )
        _seed_ref(connection, edge_event_id="event:conflict", decision_trace_id=TRACE_B)
        connection.commit()

    # When the explanation query resolves the event
    result = _query()(database).get("event:conflict")

    # Then the mismatch is a typed TRACE_REF_CONFLICT and no arbitrary trace is chosen
    assert isinstance(result, TraceRefConflict)
    assert not isinstance(result, EventExplanationFacts)
    assert result.code == "TRACE_REF_CONFLICT"
    assert result.edge_event_id == "event:conflict"
    assert result.incident_decision_trace_id == TRACE_A
    assert result.ref_decision_trace_id == TRACE_B
    assert not hasattr(result, "decision_trace_id")


def test_explanation_query_returns_none_for_missing_event(tmp_path: Path) -> None:
    # Given a migrated database with no matching evidence_events row
    database = _migrated(tmp_path)

    # When the explanation query looks up an unknown edge_event_id
    result = _query()(database).get("event:missing")

    # Then the result is None rather than an empty or fabricated fact set
    assert result is None


def test_explanation_query_keeps_missing_runtime_and_decision_values_explicit(
    tmp_path: Path,
) -> None:
    from backend.app.features.evidence.explanation_store import EventExplanationFacts

    # Given an event+incident with a decision row, no values, and no retained runtime row
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=9)
        _seed_decision(
            connection,
            trace_id=TRACE_A,
            analysis_id=ANALYSIS_A,
            values=(),
        )
        _seed_event(connection, edge_event_id="event:partial")
        _seed_incident(
            connection,
            incident_id="incident:partial",
            edge_event_id="event:partial",
            decision_trace_id=TRACE_A,
        )
        connection.execute(
            "DELETE FROM runtime_analysis_traces WHERE trace_id = ?",
            (ANALYSIS_A,),
        )
        connection.commit()

    # When the explanation query projects the event
    result = _query()(database).get("event:partial")

    # Then decision identity remains and missing runtime/values stay empty, not invented
    assert isinstance(result, EventExplanationFacts)
    assert result.decision is not None
    assert result.decision.decision_trace_id == TRACE_A
    assert result.decision.analysis_trace_id is None
    assert result.decision.values == ()
    assert result.runtime is None


def test_explanation_query_privacy_sentinel_never_appears_in_results_or_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from backend.app.features.evidence import explanation_store

    # Given a complete event whose payload_json contains a unique privacy sentinel
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest(connection)
        _seed_analysis(connection, analysis_id=ANALYSIS_A, frame_seq=5)
        _seed_decision(connection, trace_id=TRACE_A, analysis_id=ANALYSIS_A)
        _seed_event(connection, edge_event_id="event:private")
        _seed_incident(
            connection,
            incident_id="incident:private",
            edge_event_id="event:private",
            decision_trace_id=TRACE_A,
        )
        _seed_ref(connection, edge_event_id="event:private", decision_trace_id=TRACE_A)
        connection.commit()

    # When the explanation query projects the event under captured logs
    with caplog.at_level(logging.DEBUG):
        result = explanation_store.EventExplanationQuery(database).get("event:private")

    captured = capsys.readouterr()
    rendered = repr(result)

    # Then neither the fact set, logs, stdout, nor SQL source expose payload_json
    assert result is not None
    assert not hasattr(result, "payload_json")
    assert not hasattr(result, "facility_id")
    assert PRIVACY_SENTINEL not in rendered
    assert PRIVACY_SENTINEL not in caplog.text
    assert PRIVACY_SENTINEL not in captured.out
    assert PRIVACY_SENTINEL not in captured.err
    source = Path(explanation_store.__file__).read_text(encoding="utf-8")
    assert "payload_json" not in source
    assert "facility_id" not in source
    assert "COALESCE" not in source.upper()
