from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_fp_attribution_evidence import (
    ACTOR_SENTINEL,
    GEOMETRY_SENTINEL,
    NOTE_SENTINEL,
    PATH_SENTINEL,
    PAYLOAD_SENTINEL,
    _complete_seqs,
    _connect,
    _extract,
    _migrated,
    _record_for,
    _seed_fp_event,
)
from test_fp_attribution_precedence import _complete_record, _duplicate_export

from worker.fp_attribution import (
    FalsePositiveCohortQuery,
    classify_record,
    metric_event_from_record,
    open_query_only_connection,
    summarize_attribution_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "fp-attribution-report-v1"
REPORT_VERSION = 1
DB_UNAVAILABLE = "production-edge-db-unavailable"
DB_SCHEMA_INVALID = "production-edge-db-schema-invalid"
EXPORT_UNTRUSTED = "export-untrusted"
FORBIDDEN_IMPORT_PREFIXES = (
    "backend",
    "torch",
    "ultralytics",
    "worker.adapters",
    "worker.domains",
    "worker.pipeline",
    "worker.replay",
    "worker.runtime",
    "worker.__main__",
)
_FORBIDDEN_OUTPUT_TOKENS = (
    NOTE_SENTINEL,
    ACTOR_SENTINEL,
    PAYLOAD_SENTINEL,
    PATH_SENTINEL,
    GEOMETRY_SENTINEL,
    "payload_json",
    "actor_id",
    "notes",
    "rtsp://",
    "Traceback",
    "IDLE_STATIC",
    "canonical",
    "polygon",
)
_OWNED_CLI_FILES = (
    ROOT / "worker" / "fp_attribution" / "cli.py",
    ROOT / "worker" / "fp_attribution" / "__main__.py",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "worker.fp_attribution", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _seed_happy(database: Path) -> tuple[str, str, str]:
    with _connect(database) as connection:
        complete = _seed_fp_event(
            connection,
            suffix="complete",
            seqs=_complete_seqs(),
            attempt_count=3,
        )
        pruned = _seed_fp_event(
            connection,
            suffix="pruned",
            seqs=tuple(range(71, 100)),
            trigger_seq=99,
        )
        unknown = _seed_fp_event(
            connection,
            suffix="unknown",
            seqs=tuple(range(200, 230)),
            trigger_seq=229,
            values=None,
        )
        _seed_fp_event(
            connection,
            suffix="tp",
            seqs=tuple(range(300, 330)),
            trigger_seq=329,
            disposition="TRUE_POSITIVE",
        )
        connection.commit()
    return complete, pruned, unknown


def _write_probe(database: Path) -> None:
    connection = open_query_only_connection(database)
    try:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, "
            "next_attempt_at) VALUES ('event:write', '2026-08-13T12:00:00Z', "
            "'{}', 'STAGED', 1, 1)"
        )
    finally:
        connection.close()


def test_todo10_13_seams_remain_the_cli_composition_inputs(tmp_path: Path) -> None:
    """Characterize committed Todo 10-13 outputs before any CLI exists.

    Given a migrated v16 database with one current FP, one TP exclusion, and
    one COMPLETE evidence row
    When the committed cohort, evidence, classifier, and metrics seams run
    Then members stay current-FP only, evidence category stays null, one
    classified category is emitted, and alert absence stays typed unavailable.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        current_fp = _seed_fp_event(
            connection,
            suffix="compose",
            seqs=_complete_seqs(),
        )
        _seed_fp_event(
            connection,
            suffix="compose-tp",
            seqs=tuple(range(71, 101)),
            trigger_seq=100,
            disposition="TRUE_POSITIVE",
        )
        connection.commit()

    cohort = FalsePositiveCohortQuery(database).load()
    evidence = _extract(database)
    record = _record_for(evidence, current_fp)
    classified = classify_record(_complete_record(attempt_count=record.attempt_count))
    summary = summarize_attribution_metrics(
        (metric_event_from_record(record, decision=classify_record(record)),),
        exclusions=cohort.exclusions,
    )

    assert tuple(member.edge_event_id for member in cohort.members) == (current_fp,)
    assert {item.reason for item in cohort.exclusions} == {"TRUE_POSITIVE"}
    assert record.category is None
    assert record.evidence_status == "COMPLETE"
    assert classified.category == "UNCATEGORIZED"
    assert classified.annotations.attempt_count == 1
    assert summary.cohort_total == 1
    assert summary.transport.unique_alert_id.status == "UNAVAILABLE"
    assert summary.transport.unique_alert_id.value is None
    assert summary.transport.unique_alert_id.missing_reason == (
        "alert_correlation_export_not_supplied"
    )
    assert _duplicate_export()["schema"] == "fp-correlation-v1"


def test_help_exits_zero_without_database_or_bootstrap() -> None:
    """--help must work without opening a database or worker bootstrap.

    Given no --edge-db argument
    When python -m worker.fp_attribution --help runs
    Then it exits 0, documents the required flag, and writes help to stdout.
    """

    result = _run("--help")

    assert result.returncode == 0
    assert "--edge-db" in result.stdout
    assert "--correlation-export" in result.stdout
    assert "--alert-export" in result.stdout
    assert result.stdout != ""


def test_help_does_not_import_worker_model_gpu_or_replay() -> None:
    """The standalone entrypoint must not bootstrap replay, runtime, or models.

    Given the owned CLI modules
    When their imports are inspected and --help runs in a fresh interpreter
    Then worker.replay, worker.runtime, models, and GPU stacks stay unloaded.
    """

    for path in _OWNED_CLI_FILES:
        imported = _imported_modules(path)
        for name in imported:
            assert not any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            ), name

    probe = r"""
import runpy
import sys
sys.argv = ["worker.fp_attribution", "--help"]
try:
    runpy.run_module("worker.fp_attribution", run_name="__main__")
except SystemExit as exc:
    code = 0 if exc.code in (0, None) else int(exc.code)
else:
    code = 0
mods = [
    name
    for name in sys.modules
    if name == "worker.replay"
    or name.startswith("worker.replay.")
    or name == "worker.runtime"
    or name.startswith("worker.runtime.")
    or name == "worker.__main__"
    or name.startswith("torch")
    or name.startswith("ultralytics")
]
print("EXIT", code)
print("MODS", ",".join(sorted(mods)))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert "EXIT 0" in lines
    assert "MODS " in result.stdout
    assert result.stdout.strip().endswith("MODS") or result.stdout.split("MODS", 1)[1].strip() == ""


def test_happy_nonempty_cohort_emits_one_allowlisted_json_document(tmp_path: Path) -> None:
    """A complete/pruned/unknown fixture emits one privacy-safe report.

    Given a migrated database with one COMPLETE, one PRUNED, one UNKNOWN FP
    and one TP exclusion
    When the standalone module runs
    Then stdout is one deterministic JSON document with sorted records,
    null pruned/unknown categories, exact metrics, and no denylisted tokens.
    """

    database = _migrated(tmp_path)
    complete, pruned, unknown = _seed_happy(database)
    before = _digest(database)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    assert payload["schema"] == REPORT_SCHEMA
    assert payload["version"] == REPORT_VERSION
    assert payload["source"] == {"kind": "edge-sqlite", "read_mode": "query_only"}
    member_ids = [item["edge_event_id"] for item in payload["cohort"]["members"]]
    assert member_ids == sorted(member_ids)
    assert member_ids == [complete, pruned, unknown]
    assert payload["cohort"]["exclusion_census"] == {"TRUE_POSITIVE": 1}
    record_ids = [item["edge_event_id"] for item in payload["records"]]
    assert record_ids == sorted(record_ids)
    by_id = {item["edge_event_id"]: item for item in payload["records"]}
    assert by_id[complete]["evidence_status"] == "COMPLETE"
    assert by_id[complete]["category"] == "UNCATEGORIZED"
    assert by_id[pruned]["neighborhood_pruned"] is True
    assert by_id[pruned]["category"] is None
    assert by_id[unknown]["evidence_status"] == "UNKNOWN"
    assert by_id[unknown]["category"] is None
    assert payload["metrics"]["cohort_total"] == 3
    assert payload["metrics"]["attributable_count"] == 1
    assert payload["metrics"]["pruned_count"] == 1
    assert payload["metrics"]["unknown_count"] == 1
    assert payload["metrics"]["attribution_rate"]["numerator"] == 1
    assert payload["metrics"]["attribution_rate"]["denominator"] == 3
    assert payload["metrics"]["transport"]["unique_edge_event_count"] == 3
    assert payload["metrics"]["transport"]["total_attempts"] == 5
    assert payload["correlation"]["proof_export"] == "absent"
    assert payload["correlation"]["alert_ids"]["status"] == "UNAVAILABLE"
    assert payload["correlation"]["alert_ids"]["value"] is None
    rendered = result.stdout
    for token in _FORBIDDEN_OUTPUT_TOKENS:
        assert token not in rendered
        assert token not in result.stderr
    assert str(database) not in rendered
    assert _digest(database) == before


def test_empty_cohort_emits_zero_counts_and_unavailable_ratios(tmp_path: Path) -> None:
    """An empty migrated database is a successful empty report.

    Given a migrated v16 database with no incidents
    When the standalone module runs
    Then counts are zero and ratio denominators stay typed unavailable.
    """

    database = _migrated(tmp_path)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["cohort"]["members"] == []
    assert payload["cohort"]["exclusion_census"] == {}
    assert payload["records"] == []
    assert payload["metrics"]["cohort_total"] == 0
    assert payload["metrics"]["attribution_rate"]["value"] is None
    assert payload["metrics"]["attribution_rate"]["missing_reason"] == "cohort_total_zero"
    assert payload["metrics"]["attribution_coverage"]["missing_reason"] == "evaluable_total_zero"
    assert payload["correlation"]["alert_ids"]["status"] == "UNAVAILABLE"
    assert payload["correlation"]["alert_ids"]["value"] is None


def test_missing_database_exits_nonzero_without_creating_a_file(tmp_path: Path) -> None:
    """A missing --edge-db path must fail closed and create nothing.

    Given a path that does not exist
    When the standalone module runs
    Then the exit is nonzero, stdout stays empty, and no file is created.
    """

    missing = tmp_path / "missing-edge.sqlite3"

    result = _run("--edge-db", str(missing))

    assert result.returncode != 0
    assert result.stdout == ""
    assert DB_UNAVAILABLE in result.stderr
    assert "Traceback" not in result.stderr
    assert not missing.exists()
    assert not missing.with_name(f"{missing.name}-wal").exists()
    assert not missing.with_name(f"{missing.name}-shm").exists()


def test_readonly_fixture_succeeds_and_write_probe_is_denied(tmp_path: Path) -> None:
    """The CLI stays read-only even when the fixture inode is write-denied.

    Given a seeded database chmod'd read-only
    When the module runs and a test-only write probe uses the query-only seam
    Then the report succeeds and the write is denied without mutation.
    """

    database = _migrated(tmp_path)
    _seed_happy(database)
    before = _digest(database)
    database.chmod(0o444)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["cohort_total"] == 3
    with pytest.raises(sqlite3.DatabaseError, match="authorized|readonly|query_only"):
        _write_probe(database)
    assert _digest(database) == before


def test_malformed_and_untrusted_exports_fail_without_partial_stdout(tmp_path: Path) -> None:
    """Bad correlation or alert exports must not emit a partial report.

    Given a valid database plus notes-bearing, unknown-field, conflicting,
    and unreadable exports
    When the standalone module runs
    Then each case exits nonzero with empty stdout and a stable export token.
    """

    database = _migrated(tmp_path)
    complete, _, _ = _seed_happy(database)
    notes = _write_json(
        tmp_path / "notes.json",
        [
            {
                "schema": "fp-correlation-v1",
                "edge_event_id": complete,
                "kind": "DELIVERY_RETRY",
                "user_visible_delivery_count": 2,
                "notes": "operator said this retried in the UI",
            }
        ],
    )
    unknown_fields = _write_json(
        tmp_path / "unknown.json",
        [{"edge_event_id": complete, "alert_id": "alert:one", "actor_id": ACTOR_SENTINEL}],
    )
    conflict = _write_json(
        tmp_path / "conflict.json",
        [
            {
                "schema": "fp-correlation-v1",
                "edge_event_id": complete,
                "kind": "DELIVERY_RETRY",
                "user_visible_delivery_count": 2,
            },
            {
                "schema": "fp-correlation-v1",
                "edge_event_id": complete,
                "kind": "BACKEND_OR_UI_DUPLICATE",
                "user_visible_delivery_count": 3,
            },
        ],
    )
    missing_export = tmp_path / "absent-export.json"

    cases = (
        ("--correlation-export", notes),
        ("--alert-export", unknown_fields),
        ("--correlation-export", conflict),
        ("--alert-export", missing_export),
    )
    for flag, export in cases:
        result = _run("--edge-db", str(database), flag, str(export))
        assert result.returncode != 0
        assert result.stdout == ""
        assert EXPORT_UNTRUSTED in result.stderr
        assert "Traceback" not in result.stderr
        assert NOTE_SENTINEL not in result.stderr
        assert ACTOR_SENTINEL not in result.stderr
        assert not missing_export.exists()


def test_absent_export_stays_unavailable_and_empty_alert_export_is_zero(
    tmp_path: Path,
) -> None:
    """Alert absence is unavailable; an explicit empty list is available zero.

    Given the same happy database
    When the module runs without an export and again with an empty alert list
    Then unique_alert_id is UNAVAILABLE, then AVAILABLE value 0.
    """

    database = _migrated(tmp_path)
    _seed_happy(database)
    empty = _write_json(tmp_path / "alerts.json", [])

    absent = _run("--edge-db", str(database))
    present = _run("--edge-db", str(database), "--alert-export", str(empty))

    assert absent.returncode == 0
    assert present.returncode == 0
    absent_payload = json.loads(absent.stdout)
    present_payload = json.loads(present.stdout)
    assert absent_payload["correlation"]["alert_ids"]["status"] == "UNAVAILABLE"
    assert absent_payload["correlation"]["alert_ids"]["value"] is None
    assert present_payload["correlation"]["alert_ids"]["status"] == "AVAILABLE"
    assert present_payload["correlation"]["alert_ids"]["value"] == 0
    assert present_payload["correlation"]["alert_ids"]["missing_reason"] is None
    assert present_payload["metrics"]["transport"]["unique_edge_event_count"] == 3


def test_repeated_invocations_emit_byte_identical_stdout(tmp_path: Path) -> None:
    """Identical inputs must not drift across invocations.

    Given one seeded database
    When the standalone module runs twice
    Then the stdout bytes are identical and stderr stays empty.
    """

    database = _migrated(tmp_path)
    _seed_happy(database)

    first = _run("--edge-db", str(database))
    second = _run("--edge-db", str(database))

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stderr == ""
    assert second.stderr == ""
    assert first.stdout.encode("utf-8") == second.stdout.encode("utf-8")


def test_errors_go_only_to_stderr_and_success_keeps_streams_separated(
    tmp_path: Path,
) -> None:
    """Success JSON and failure tokens stay on opposite streams.

    Given one valid database and one missing path
    When both invocations run
    Then success has empty stderr and failure has empty stdout.
    """

    database = _migrated(tmp_path)
    _seed_happy(database)
    missing = tmp_path / "no-such.sqlite3"

    success = _run("--edge-db", str(database))
    failure = _run("--edge-db", str(missing))

    assert success.returncode == 0
    assert success.stderr == ""
    json.loads(success.stdout)
    assert failure.returncode != 0
    assert failure.stdout == ""
    assert failure.stderr != ""
    assert DB_UNAVAILABLE in failure.stderr


def test_unreadable_schema_uses_stable_schema_token(tmp_path: Path) -> None:
    """A non-v16 sqlite file is a schema error, not a fabricated cohort.

    Given an empty sqlite file that is not schema v16
    When the standalone module runs
    Then stdout stays empty and the schema token is on stderr.
    """

    database = tmp_path / "not-v16.sqlite3"
    sqlite3.connect(database).close()

    result = _run("--edge-db", str(database))

    assert result.returncode != 0
    assert result.stdout == ""
    assert DB_SCHEMA_INVALID in result.stderr
    assert "Traceback" not in result.stderr
    assert database.is_file()


def _selected_rtsp_sentinel() -> str:
    return "".join(
        (
            "rtsp",
            "://",
            "user",
            ":",
            "CLI_pass_9e44",
            "@",
            "10.255.255.4",
            "/stream",
        )
    )


def test_schema_valid_selected_reason_and_state_text_never_crosses_cli(
    tmp_path: Path,
) -> None:
    """Schema-valid selected decision text must not leak through CLI JSON/streams.

    Given a current FP whose reason/previous_state/current_state are a
    runtime-composed credentialed RTSP value
    When the standalone module runs
    Then the exact secret never appears in JSON, stdout, or stderr, and the
    record stays typed unavailable rather than classified from untrusted text.
    """

    secret = _selected_rtsp_sentinel()
    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="cli-poisoned",
            seqs=_complete_seqs(),
        )
        updated = connection.execute(
            "UPDATE evidence_decision_traces "
            "SET reason = ?, previous_state = ?, current_state = ?",
            (secret, secret, secret),
        ).rowcount
        assert updated == 1
        connection.commit()
    before = _digest(database)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    records = {
        item["edge_event_id"]: item for item in payload["records"]
    }
    record = records[edge_event_id]
    assert record["decision_reason"] is None
    assert record["previous_state"] is None
    assert record["current_state"] is None
    assert record["evidence_status"] == "UNKNOWN"
    assert record["category"] is None
    assert record["neighborhood_pruned"] is False
    assert secret not in result.stdout
    assert secret not in result.stderr
    for token in _FORBIDDEN_OUTPUT_TOKENS:
        assert token not in result.stdout
        assert token not in result.stderr
    assert _digest(database) == before
    leaked = json.dumps({"decision_reason": secret, "previous_state": secret})
    with pytest.raises(AssertionError):
        assert secret not in leaked
        assert secret not in result.stdout


@pytest.mark.parametrize(
    "secret",
    (
        "/private/cli-reason-path.bin",
        "ghp_" + ("C" * 36),
        "w" * 257,
        "fall-onset\x07",
        "f\u0430ll-onset",
    ),
    ids=(
        "absolute_path",
        "token_like",
        "overlength",
        "control_chars",
        "unicode_confusable",
    ),
)
def test_hostile_selected_reason_and_state_text_stays_unknown_on_cli(
    tmp_path: Path,
    secret: str,
) -> None:
    """Residual hostile selected reason/state values stay typed unavailable.

    Given a current FP whose selected reason/state columns are schema-valid
    hostile text from the verifier residual classes
    When the standalone module runs
    Then the exact secret never appears and the record stays UNKNOWN.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="cli-hostile",
            seqs=_complete_seqs(),
        )
        updated = connection.execute(
            "UPDATE evidence_decision_traces "
            "SET reason = ?, previous_state = ?, current_state = ?",
            (secret, secret, secret),
        ).rowcount
        assert updated == 1
        connection.commit()
    before = _digest(database)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    record = next(
        item for item in payload["records"] if item["edge_event_id"] == edge_event_id
    )
    assert record["decision_reason"] is None
    assert record["previous_state"] is None
    assert record["current_state"] is None
    assert record["evidence_status"] == "UNKNOWN"
    assert record["category"] is None
    assert record["neighborhood_pruned"] is False
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert result.stderr == ""
    assert _digest(database) == before
    leaked = {"decision_reason": secret}
    with pytest.raises(AssertionError):
        assert leaked["decision_reason"] is None
        assert record["decision_reason"] is None


def test_cli_serializes_typed_domain_evidence_without_raw_db_payload(
    tmp_path: Path,
) -> None:
    """CLI JSON must emit stable domain facts and never the raw DB payload.

    Given an aligned fall-latch fixture with person-gap and non-due pose rows
    When the standalone module runs
    Then typed domain evidence appears with stable machine fields, category is
    unchanged, and payload/notes/path/geometry never appear.
    """

    from test_fp_attribution_evidence import _pose_components

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="cli-domain",
            seqs=seqs,
            analysis_track_by_seq={seq: None if seq >= 36 else 7 for seq in seqs},
            components_by_seq={
                seq: _pose_components("not-scheduled" if seq in {20, 21} else "observed")
                for seq in seqs
            },
            neighborhood_decisions=(
                {
                    "seq": 30,
                    "reason": "fall-onset",
                    "previous_state": "clear",
                    "current_state": "fall",
                    "triggered": 1,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.88, None),),
                },
                {
                    "seq": 33,
                    "reason": "below-threshold",
                    "previous_state": "fall",
                    "current_state": "clear",
                    "triggered": 0,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.12, None),),
                },
            ),
        )
        connection.commit()
    before = _digest(database)

    result = _run("--edge-db", str(database))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    record = next(
        item for item in payload["records"] if item["edge_event_id"] == edge_event_id
    )
    assert record["category"] == "UNCATEGORIZED"
    assert record["person_presence"] == {
        "duration_frames": 4,
        "missing_reason": None,
        "status": "PERSON_GAP",
    }
    assert record["due_signal"] == {
        "missing_reason": None,
        "not_scheduled_frames": 2,
        "status": "NOT_DUE",
    }
    assert record["fall_latch"] == {
        "missing_reason": None,
        "rearm_frames": 7,
        "rise_before_rearm": True,
        "same_domain": True,
        "same_track": True,
        "status": "AVAILABLE",
    }
    assert record["bed_state"]["status"] == "NOT_APPLICABLE"
    assert record["track_staleness"]["last_seen_offset_frames"] == 4
    assert record["domain_alignment"]["domain"] == "fall"
    assert record["domain_alignment"]["status"] == "ALIGNED"
    assert record["boot_changed"] is False
    assert record["epoch_changed"] is False
    assert "payload_json" not in record
    assert "payload_json" not in result.stdout
    for token in _FORBIDDEN_OUTPUT_TOKENS:
        assert token not in result.stdout
        assert token not in result.stderr
    assert _digest(database) == before
