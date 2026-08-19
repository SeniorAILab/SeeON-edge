from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import pytest

from shared.edge_db.migrator import migrate_database
from shared.edge_db.review_migration import classify_legacy_labels
from shared.edge_db.schema import SCHEMA_VERSION

NOW = "2026-08-13T12:00:00Z"
LATER = "2026-08-13T12:01:00Z"
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
NOTE_SENTINEL = "NOTE_SENTINEL_fp_attribution_7c21"
ACTOR_SENTINEL = "actor:sentinel-fp-attribution"
PAYLOAD_SENTINEL = "PAYLOAD_SENTINEL_fp_json_9e44"
PATH_SENTINEL = "/tmp/seeon-forbidden/clip.mp4"
GEOMETRY_SENTINEL = "polygon:[[1.25,9.5],[3.5,8.25]]"

_FORBIDDEN_SOURCE_TOKENS = (
    "notes",
    "actor_id",
    "payload_json",
    "contained_relpath",
    "media_relpath",
    "manifest_path",
    "polygon",
    "geometry",
    "x1",
    "y1",
    "COALESCE",
)


def _trace_id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _migrated(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _seed_event(
    connection: sqlite3.Connection,
    *,
    edge_event_id: str,
    payload_json: str | None = None,
) -> None:
    payload = payload_json
    if payload is None:
        payload = (
            f'{{"secret":"{PAYLOAD_SENTINEL}","path":"{PATH_SENTINEL}",'
            f'"geometry":"{GEOMETRY_SENTINEL}"}}'
        )
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
        "VALUES (?, ?, ?, 'STAGED', 1, 1)",
        (edge_event_id, NOW, payload),
    )


def _seed_clip(connection: sqlite3.Connection, clip_id: str) -> None:
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, state_version) "
        "VALUES (?, 'VERIFIED', 1)",
        (clip_id,),
    )


def _seed_incident(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    edge_event_id: str,
    clip_id: str | None,
    event_type: str = "fall",
    decision_trace_id: str | None = None,
) -> None:
    if decision_trace_id is None:
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, primary_clip_id, lifecycle_state,
                created_at, updated_at
            ) VALUES (?, ?, 'camera:opaque', ?, ?, 'NOT_RECORDED', ?,
                      'STAGING', ?, ?)
            """,
            (incident_id, edge_event_id, event_type, NOW, clip_id, NOW, NOW),
        )
        return
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            runtime_manifest_sha256, decision_trace_id, module_qualified_id,
            policy_qualified_id, effective_policy_id, provenance_state,
            primary_clip_id, lifecycle_state, created_at, updated_at
        ) VALUES (?, ?, 'camera:opaque', ?, ?, ?, ?, 'fall.v1', 'fall.policy.v1',
                  ?, 'QUALIFIED', ?, 'STAGING', ?, ?)
        """,
        (
            incident_id,
            edge_event_id,
            event_type,
            NOW,
            MANIFEST_ID,
            decision_trace_id,
            POLICY_ID,
            clip_id,
            NOW,
            NOW,
        ),
    )


def _seed_primary(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    clip_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, 'NOT_RECORDED', '[]', 'MISSING', ?)
        """,
        (incident_id, clip_id, NOW),
    )


def _seed_review(
    connection: sqlite3.Connection,
    *,
    review_id: str,
    incident_id: str,
    clip_id: str,
    version: int,
    disposition: str,
    reviewed_at: str = NOW,
    notes: str | None = NOTE_SENTINEL,
    actor_id: str = ACTOR_SENTINEL,
) -> None:
    connection.execute(
        """
        INSERT INTO control_evidence_review_revisions (
            review_id, incident_id, clip_id, review_version, actor_id,
            reviewed_at, disposition, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            incident_id,
            clip_id,
            version,
            actor_id,
            reviewed_at,
            disposition,
            notes,
        ),
    )


def _seed_review_state(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    clip_id: str,
    current_version: int,
) -> None:
    existing = connection.execute(
        "SELECT current_version FROM control_evidence_review_state WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO control_evidence_review_state "
            "(incident_id, clip_id, current_version) VALUES (?, ?, ?)",
            (incident_id, clip_id, current_version),
        )
        return
    connection.execute(
        "UPDATE control_evidence_review_state SET current_version = ? WHERE incident_id = ?",
        (current_version, incident_id),
    )


def _seed_current_review(
    connection: sqlite3.Connection,
    *,
    incident_id: str,
    clip_id: str,
    disposition: str,
    versions: tuple[str, ...],
) -> None:
    for index, disposition_at_version in enumerate(versions, start=1):
        _seed_review(
            connection,
            review_id=f"review:{incident_id}:{index}",
            incident_id=incident_id,
            clip_id=clip_id,
            version=index,
            disposition=disposition_at_version,
            reviewed_at=NOW if index == 1 else LATER,
        )
        _seed_review_state(
            connection,
            incident_id=incident_id,
            clip_id=clip_id,
            current_version=index,
        )
    assert versions[-1] == disposition


def _seed_case(
    connection: sqlite3.Connection,
    *,
    suffix: str,
    disposition: str | None,
    versions: tuple[str, ...] | None = None,
    event_type: str = "fall",
    decision_trace_id: str | None = None,
    ref_trace_id: str | None = None,
    clip: bool = True,
) -> tuple[str, str, str | None]:
    edge_event_id = f"event:{suffix}"
    incident_id = f"incident:{suffix}"
    clip_id = f"clip:{suffix}" if clip else None
    _seed_event(connection, edge_event_id=edge_event_id)
    if clip_id is not None:
        _seed_clip(connection, clip_id)
    _seed_incident(
        connection,
        incident_id=incident_id,
        edge_event_id=edge_event_id,
        clip_id=clip_id,
        event_type=event_type,
        decision_trace_id=decision_trace_id,
    )
    if clip_id is not None:
        _seed_primary(connection, incident_id=incident_id, clip_id=clip_id)
    if ref_trace_id is not None:
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
            (edge_event_id, ref_trace_id),
        )
    if disposition is not None:
        history = versions if versions is not None else (disposition,)
        _seed_current_review(
            connection,
            incident_id=incident_id,
            clip_id=clip_id or f"clip:{suffix}",
            disposition=disposition,
            versions=history,
        )
    return edge_event_id, incident_id, clip_id


def _seed_manifest_and_traces(
    connection: sqlite3.Connection,
    *,
    decision_trace_id: str,
    analysis_trace_id: str,
) -> None:
    connection.execute(
        "INSERT INTO runtime_manifest_contents VALUES (?, 1, '{}', ?)",
        (MANIFEST_ID, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_boots VALUES ('boot-a', ?, ?)",
        (MANIFEST_ID, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_manifest_cameras VALUES ('boot-a', 'camera:opaque', ?, ?)",
        (MANIFEST_ID, NOW),
    )
    connection.execute(
        "INSERT INTO runtime_analysis_traces "
        "(trace_id, trace_schema_version, worker_boot_id, camera_id, stream_epoch, "
        "frame_seq, pts, source_time_sec, frame_width, frame_height, "
        "bed_region_provenance, storage_bytes) "
        "VALUES (?, 1, 'boot-a', 'camera:opaque', 1, 12, 0, 0, 320, 180, 'fresh', 1)",
        (analysis_trace_id,),
    )
    connection.execute(
        "INSERT INTO evidence_decision_traces "
        "(trace_id, trace_schema_version, analysis_trace_id, module_qualified_id, "
        "policy_qualified_id, effective_policy_id, runtime_manifest_sha256, reason, "
        "previous_state, current_state, triggered, track_id, track_missing_reason, "
        "bed_id, bed_missing_reason) "
        "VALUES (?, 1, ?, 'fall.v1', 'fall.policy.v1', ?, ?, 'fall-onset', "
        "'clear', 'fall', 1, 7, NULL, NULL, 'not-applicable')",
        (decision_trace_id, analysis_trace_id, POLICY_ID, MANIFEST_ID),
    )


def _load_cohort(database: Path):
    from worker.fp_attribution import FalsePositiveCohortQuery

    return FalsePositiveCohortQuery(database).load()


def _exclusion_census(result: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.exclusions:
        reason = str(item.reason)
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _member_ids(result: object) -> tuple[str, ...]:
    return tuple(member.edge_event_id for member in result.members)


def test_schema_v16_current_review_revision_is_the_live_row(tmp_path: Path) -> None:
    # Given a migrated v16 incident with an older FP revision and a current TP
    assert SCHEMA_VERSION == 16
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_case(
            connection,
            suffix="current-revision",
            disposition="TRUE_POSITIVE",
            versions=("FALSE_POSITIVE", "TRUE_POSITIVE"),
        )
        connection.commit()
        current = connection.execute(
            """
            SELECT revision.review_version, revision.disposition
            FROM control_evidence_review_state AS current
            JOIN control_evidence_review_revisions AS revision
              ON revision.incident_id = current.incident_id
             AND revision.clip_id = current.clip_id
             AND revision.review_version = current.current_version
            """
        ).fetchall()
        history = connection.execute(
            "SELECT review_version, disposition "
            "FROM control_evidence_review_revisions ORDER BY review_version"
        ).fetchall()

    # Then only version 2 is current, while the older FP revision remains stored
    assert current == [(2, "TRUE_POSITIVE")]
    assert history == [(1, "FALSE_POSITIVE"), (2, "TRUE_POSITIVE")]


def test_legacy_label_migration_classifies_unique_ambiguous_orphan_and_unsupported(
    tmp_path: Path,
) -> None:
    # Given unique, ambiguous, orphan, and unsupported leftover labels on v16
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_case(connection, suffix="unique-legacy", disposition=None)
        _seed_event(connection, edge_event_id="event:amb-a")
        _seed_event(connection, edge_event_id="event:amb-b")
        _seed_clip(connection, "clip:ambiguous")
        _seed_incident(
            connection,
            incident_id="incident:amb-a",
            edge_event_id="event:amb-a",
            clip_id="clip:ambiguous",
        )
        _seed_incident(
            connection,
            incident_id="incident:amb-b",
            edge_event_id="event:amb-b",
            clip_id="clip:ambiguous",
        )
        _seed_primary(connection, incident_id="incident:amb-a", clip_id="clip:ambiguous")
        _seed_primary(connection, incident_id="incident:amb-b", clip_id="clip:ambiguous")
        _seed_clip(connection, "clip:orphan-incident")
        labels = (
            ("clip:unique-legacy", "FALSE_POSITIVE", "legacy:operator", NOW),
            ("clip:ambiguous", "FALSE_POSITIVE", "legacy:operator", NOW),
            ("clip:orphan-incident", "FALSE_POSITIVE", "legacy:operator", NOW),
            ("clip:missing", "FALSE_POSITIVE", "legacy:operator", NOW),
            ("clip:unsupported", None, "legacy:operator", NOW),
        )
        for clip_id, label, reviewer, reviewed_at in labels:
            connection.execute(
                "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
                "VALUES (?, ?, ?, ?, '{}')",
                (clip_id, label, reviewer, reviewed_at),
            )
        connection.commit()
        classify_legacy_labels(connection)
        migrated = connection.execute(
            "SELECT incident_id, clip_id, review_version, disposition "
            "FROM control_evidence_review_revisions"
        ).fetchall()
        classifications = dict(
            connection.execute(
                "SELECT source_clip_id, classification FROM control_legacy_label_migrations"
            ).fetchall()
        )

    # Then only the unique clip-to-incident label becomes a current review
    assert migrated == [("incident:unique-legacy", "clip:unique-legacy", 1, "FALSE_POSITIVE")]
    assert classifications == {
        "clip:unique-legacy": "MIGRATED",
        "clip:ambiguous": "AMBIGUOUS_INCIDENT",
        "clip:orphan-incident": "ORPHAN_INCIDENT",
        "clip:missing": "ORPHAN_CLIP",
        "clip:unsupported": "UNSUPPORTED_DISPOSITION",
    }


def test_current_fp_enters_cohort_and_older_fp_revision_does_not(tmp_path: Path) -> None:
    # Given a current FP that replaced an older TP, and a current TP that replaced an FP
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        current_fp, _, _ = _seed_case(
            connection,
            suffix="now-fp",
            disposition="FALSE_POSITIVE",
            versions=("TRUE_POSITIVE", "FALSE_POSITIVE"),
        )
        older_fp, _, _ = _seed_case(
            connection,
            suffix="was-fp",
            disposition="TRUE_POSITIVE",
            versions=("FALSE_POSITIVE", "TRUE_POSITIVE"),
        )
        connection.commit()

    # When the read-only cohort is loaded
    result = _load_cohort(database)

    # Then only the current FP revision's event is eligible
    assert _member_ids(result) == (current_fp,)
    assert result.members[0].current_review_version == 2
    assert older_fp not in _member_ids(result)
    assert _exclusion_census(result)["TRUE_POSITIVE"] == 1


def test_true_positive_and_unreviewed_rows_are_typed_census_exclusions(
    tmp_path: Path,
) -> None:
    # Given one current FP, one current TP, and one unreviewed incident
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        current_fp, _, _ = _seed_case(
            connection,
            suffix="fp",
            disposition="FALSE_POSITIVE",
        )
        current_tp, _, _ = _seed_case(
            connection,
            suffix="tp",
            disposition="TRUE_POSITIVE",
        )
        unreviewed, _, _ = _seed_case(
            connection,
            suffix="open",
            disposition=None,
        )
        connection.commit()

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then TP and unreviewed stay outside the cohort and outside coverage denominators
    assert _member_ids(result) == (current_fp,)
    assert current_tp not in _member_ids(result)
    assert unreviewed not in _member_ids(result)
    assert _exclusion_census(result) == {
        "TRUE_POSITIVE": 1,
        "UNREVIEWED": 1,
    }


def test_duplicate_review_revisions_collapse_to_one_edge_event(tmp_path: Path) -> None:
    # Given two immutable FP revisions for one incident
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        edge_event_id, _, _ = _seed_case(
            connection,
            suffix="dup-rev",
            disposition="FALSE_POSITIVE",
            versions=("FALSE_POSITIVE", "FALSE_POSITIVE"),
        )
        connection.commit()

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then review history collapses to one current edge_event_id
    assert _member_ids(result) == (edge_event_id,)
    assert result.members[0].current_review_version == 2
    assert len(result.members) == 1


def test_unique_migrated_legacy_false_positive_enters_cohort(tmp_path: Path) -> None:
    # Given a uniquely mapped migrated FALSE_POSITIVE label
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        edge_event_id, _, clip_id = _seed_case(
            connection,
            suffix="migrated-fp",
            disposition=None,
        )
        connection.execute(
            "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
            "VALUES (?, 'FALSE_POSITIVE', 'legacy:operator', ?, '{}')",
            (clip_id, NOW),
        )
        connection.commit()
        classify_legacy_labels(connection)

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then the unique MIGRATED mapping is one current FP event
    assert _member_ids(result) == (edge_event_id,)
    assert result.members[0].current_review_version == 1
    assert "MIGRATED" not in _exclusion_census(result)


def test_ambiguous_orphan_unsupported_and_unmappable_legacy_are_exclusions(
    tmp_path: Path,
) -> None:
    # Given ambiguous, orphan, unsupported, and leftover unclassified labels
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_event(connection, edge_event_id="event:amb-a")
        _seed_event(connection, edge_event_id="event:amb-b")
        _seed_clip(connection, "clip:ambiguous")
        _seed_incident(
            connection,
            incident_id="incident:amb-a",
            edge_event_id="event:amb-a",
            clip_id="clip:ambiguous",
        )
        _seed_incident(
            connection,
            incident_id="incident:amb-b",
            edge_event_id="event:amb-b",
            clip_id="clip:ambiguous",
        )
        _seed_primary(connection, incident_id="incident:amb-a", clip_id="clip:ambiguous")
        _seed_primary(connection, incident_id="incident:amb-b", clip_id="clip:ambiguous")
        _seed_clip(connection, "clip:orphan-incident")
        for clip_id, label in (
            ("clip:ambiguous", "FALSE_POSITIVE"),
            ("clip:orphan-incident", "FALSE_POSITIVE"),
            ("clip:missing", "FALSE_POSITIVE"),
            ("clip:unsupported", None),
        ):
            connection.execute(
                "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
                "VALUES (?, ?, 'legacy:operator', ?, '{}')",
                (clip_id, label, NOW),
            )
        connection.commit()
        classify_legacy_labels(connection)
        with sqlite3.connect(database) as leftover:
            leftover.execute(
                "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
                "VALUES ('clip:leftover', 'FALSE_POSITIVE', 'legacy:operator', ?, '{}')",
                (NOW,),
            )

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then no unmappable or non-unique legacy row enters the cohort
    assert _member_ids(result) == ()
    census = _exclusion_census(result)
    assert census["AMBIGUOUS_INCIDENT"] == 1
    assert census["ORPHAN_INCIDENT"] == 1
    assert census["ORPHAN_CLIP"] == 1
    assert census["UNSUPPORTED_DISPOSITION"] == 1
    assert census["UNMAPPABLE_LEGACY"] == 1
    assert census["UNREVIEWED"] == 2
    assert "event:amb-a" not in _member_ids(result)
    assert "event:amb-b" not in _member_ids(result)


def test_operator_only_leftover_and_trace_cardinality_conflict_are_exclusions(
    tmp_path: Path,
) -> None:
    # Given a SYSTEM_TEST leftover FP and a current FP with unequal trace refs
    database = _migrated(tmp_path)
    analysis_a = _trace_id("analysis-a")
    analysis_b = _trace_id("analysis-b")
    trace_a = _trace_id("trace-a")
    trace_b = _trace_id("trace-b")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_manifest_and_traces(
            connection,
            decision_trace_id=trace_a,
            analysis_trace_id=analysis_a,
        )
        connection.execute(
            "INSERT INTO runtime_analysis_traces "
            "(trace_id, trace_schema_version, worker_boot_id, camera_id, stream_epoch, "
            "frame_seq, pts, source_time_sec, frame_width, frame_height, "
            "bed_region_provenance, storage_bytes) "
            "VALUES (?, 1, 'boot-a', 'camera:opaque', 1, 13, 0, 0, 320, 180, 'fresh', 1)",
            (analysis_b,),
        )
        connection.execute(
            "INSERT INTO evidence_decision_traces "
            "(trace_id, trace_schema_version, analysis_trace_id, module_qualified_id, "
            "policy_qualified_id, effective_policy_id, runtime_manifest_sha256, reason, "
            "previous_state, current_state, triggered, track_id, track_missing_reason, "
            "bed_id, bed_missing_reason) "
            "VALUES (?, 1, ?, 'fall.v1', 'fall.policy.v1', ?, ?, 'fall-onset', "
            "'clear', 'fall', 1, 8, NULL, NULL, 'not-applicable')",
            (trace_b, analysis_b, POLICY_ID, MANIFEST_ID),
        )
        leftover, _, _ = _seed_case(
            connection,
            suffix="system-test",
            disposition="FALSE_POSITIVE",
            event_type="SYSTEM_TEST",
        )
        conflict, _, _ = _seed_case(
            connection,
            suffix="conflict",
            disposition="FALSE_POSITIVE",
            decision_trace_id=trace_a,
            ref_trace_id=trace_b,
        )
        connection.commit()

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then leftovers and cardinality conflicts stay typed exclusions
    assert _member_ids(result) == ()
    census = _exclusion_census(result)
    assert census["OPERATOR_ONLY_LEFTOVER"] == 1
    assert census["TRACE_REF_CONFLICT"] == 1
    assert leftover not in _member_ids(result)
    assert conflict not in _member_ids(result)


def test_duplicate_event_identity_collapses_and_absent_trace_stays_in_cohort(
    tmp_path: Path,
) -> None:
    # Given a uniquely migrated current FP whose incident stores no decision trace
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        edge_event_id, _, clip_id = _seed_case(
            connection,
            suffix="no-trace",
            disposition=None,
        )
        connection.execute(
            "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
            "VALUES (?, 'FALSE_POSITIVE', 'legacy:operator', ?, '{}')",
            (clip_id, NOW),
        )
        connection.commit()
        classify_legacy_labels(connection)

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then migrated current state and the label mapping collapse to one event
    assert _member_ids(result) == (edge_event_id,)
    assert result.members[0].decision_trace_id is None
    assert result.members[0].incident_id == "incident:no-trace"


def test_zero_cohort_is_empty_members_and_empty_exclusion_census(tmp_path: Path) -> None:
    # Given a migrated v16 database with no incidents, reviews, or labels
    database = _migrated(tmp_path)

    # When the cohort is loaded
    result = _load_cohort(database)

    # Then both the cohort and the typed exclusion census are empty
    assert _member_ids(result) == ()
    assert result.exclusions == ()
    assert _exclusion_census(result) == {}


def test_query_only_authorizer_rejects_writes(tmp_path: Path) -> None:
    # Given a query-only connection over a real migrated database
    database = _migrated(tmp_path)
    from worker.fp_attribution import open_query_only_connection

    connection = open_query_only_connection(database)
    try:
        # When a write probe runs through that seam
        with pytest.raises(sqlite3.DatabaseError, match="authorized|readonly|query_only"):
            connection.execute(
                "INSERT INTO evidence_events "
                "(edge_event_id, detected_at, payload_json, state, queued_at, "
                "next_attempt_at) VALUES ('event:write', ?, '{}', 'STAGED', 1, 1)",
                (NOW,),
            )
        with pytest.raises(sqlite3.DatabaseError, match="authorized|readonly|query_only"):
            connection.execute("DELETE FROM evidence_events")
    finally:
        connection.close()

    with sqlite3.connect(database) as verify:
        # Then the database contents are unchanged
        assert verify.execute("SELECT count(*) FROM evidence_events").fetchone() == (0,)


def test_cohort_rejects_malformed_database_path(tmp_path: Path) -> None:
    from worker.fp_attribution import FalsePositiveCohortQuery

    # Given malformed or missing database identities
    query = FalsePositiveCohortQuery(tmp_path / "missing.sqlite3")

    # When the caller asks for a cohort
    # Then the seam rejects the input before any row projection
    with pytest.raises(ValueError, match="edge-db"):
        query.load()
    with pytest.raises(ValueError, match="edge-db"):
        FalsePositiveCohortQuery(Path("")).load()


def test_cohort_projection_never_exposes_forbidden_review_or_payload_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from worker.fp_attribution import cohort as cohort_module

    # Given a current FP whose review notes, actor, and payload hold sentinels
    database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _seed_case(connection, suffix="private", disposition="FALSE_POSITIVE")
        connection.commit()

    # When the cohort is loaded under captured logs
    with caplog.at_level(logging.DEBUG):
        result = _load_cohort(database)

    captured = capsys.readouterr()
    rendered = repr(result)
    source = Path(cohort_module.__file__).read_text(encoding="utf-8")

    # Then forbidden columns and sentinels never appear in output, logs, or SQL
    assert _member_ids(result) == ("event:private",)
    assert not hasattr(result.members[0], "notes")
    assert not hasattr(result.members[0], "actor_id")
    assert not hasattr(result.members[0], "payload_json")
    for token in (
        NOTE_SENTINEL,
        ACTOR_SENTINEL,
        PAYLOAD_SENTINEL,
        PATH_SENTINEL,
        GEOMETRY_SENTINEL,
    ):
        assert token not in rendered
        assert token not in caplog.text
        assert token not in captured.out
        assert token not in captured.err
    for token in _FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
        assert token.lower() not in source.lower() if token == "COALESCE" else True
    assert "COALESCE" not in source.upper()
