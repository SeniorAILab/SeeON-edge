from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
from test_fp_attribution_cli import _seed_happy
from test_fp_attribution_evidence import (
    ACTOR_SENTINEL,
    LATER,
    NOTE_SENTINEL,
    _complete_seqs,
    _connect,
    _migrated,
    _seed_fp_event,
)

from worker.fp_attribution import FalsePositiveCohortQuery
from worker.fp_attribution.cli import CLEAN_EXIT_CODE, DB_UNAVAILABLE_EXIT_CODE, main

NOW_REVIEW = LATER


def _install_connect_counter(monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_connect = sqlite3.connect

    def wrapped(*args: object, **kwargs: object) -> sqlite3.Connection:
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", wrapped)
    for name in (
        "worker.fp_attribution.cohort",
        "worker.fp_attribution.evidence",
        "worker.fp_attribution.cli",
        "worker.fp_attribution.snapshot",
    ):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "sqlite3"):
            monkeypatch.setattr(module.sqlite3, "connect", wrapped)
    return calls


def _assert_wal(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def _commit_changed_review_and_attempt(
    database: Path,
    *,
    incident_id: str,
    clip_id: str,
    edge_event_id: str,
    version: int = 2,
    attempt_count: int = 99,
) -> None:
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA foreign_keys = ON")
        assert writer.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        writer.execute(
            """
            INSERT INTO control_evidence_review_revisions (
                review_id, incident_id, clip_id, review_version, actor_id,
                reviewed_at, disposition, notes
            ) VALUES (?, ?, ?, ?, ?, ?, 'TRUE_POSITIVE', ?)
            """,
            (
                f"review:{incident_id}:{version}",
                incident_id,
                clip_id,
                version,
                ACTOR_SENTINEL,
                NOW_REVIEW,
                NOTE_SENTINEL,
            ),
        )
        writer.execute(
            "UPDATE control_evidence_review_state SET current_version = ? "
            "WHERE incident_id = ?",
            (version, incident_id),
        )
        writer.execute(
            "UPDATE evidence_events SET attempt_count = ? WHERE edge_event_id = ?",
            (attempt_count, edge_event_id),
        )
        writer.commit()
    finally:
        writer.close()


class _PinObserverConnection(sqlite3.Connection):
    """Connection subclass that fires a hook right after the pin statement.

    SQLite's BEGIN is DEFERRED: the WAL read snapshot is only pinned at the
    first statement that touches a real table, not at BEGIN itself. This
    class detects the pin statement structurally -- the first ``execute()``
    call that runs immediately after a literal ``BEGIN`` on this connection
    -- and runs a caller-supplied hook right after it returns, before any
    later statement can run. Detecting the pin this way (by position, not by
    matching a specific implementation's SQL text) means the same harness
    catches both a correct pin (e.g. a real table read) and a defective one
    (e.g. a constant-only ``SELECT 1`` that touches no table and pins
    nothing): whichever statement immediately follows BEGIN is exercised,
    and the writer commits right after it, at the exact boundary the
    isolation guarantee depends on.
    """

    on_pin: object = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._after_begin = False

    def execute(self, sql, *args, **kwargs):
        result = super().execute(sql, *args, **kwargs)
        if self._after_begin:
            self._after_begin = False
            hook = type(self).on_pin
            if hook is not None:
                type(self).on_pin = None
                hook()
        elif sql.strip().upper() == "BEGIN":
            self._after_begin = True
        return result


def _install_pin_hook(monkeypatch, module, hook) -> None:
    """Inject a connection factory so the pin statement triggers ``hook``."""

    real_connect = module.sqlite3.connect
    _PinObserverConnection.on_pin = hook

    def connect_with_factory(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs.setdefault("factory", _PinObserverConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(module.sqlite3, "connect", connect_with_factory)


def _seed_race_fixture(tmp_path: Path, *, suffix: str) -> tuple[Path, str]:
    database = _migrated(tmp_path)
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix=suffix,
            seqs=_complete_seqs(),
            attempt_count=3,
        )
        connection.commit()
    _assert_wal(database)
    return database, edge_event_id


def test_cli_keeps_one_snapshot_when_writer_commits_after_cohort(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """One read snapshot must span cohort, evidence, classification, and metrics.

    Given a WAL v16 database with one current FP and attempt_count=3
    When a second writer commits TP + attempt_count=99 immediately after the
    snapshot's pin statement executes (the true isolation boundary -- not
    merely after the cohort query, which would already have pinned the
    snapshot on its own and mask a broken pin)
    Then the CLI report still sees the pre-change FP fact, while a fresh
    connection observes the committed TP.
    """

    from worker.fp_attribution import snapshot as snapshot_mod

    database, edge_event_id = _seed_race_fixture(tmp_path, suffix="snap")

    def commit_writer_now() -> None:
        _commit_changed_review_and_attempt(
            database,
            incident_id="incident:snap",
            clip_id="clip:snap",
            edge_event_id=edge_event_id,
        )

    _install_pin_hook(monkeypatch, snapshot_mod, commit_writer_now)

    code = main(["--edge-db", str(database)])
    captured = capsys.readouterr()

    assert code == CLEAN_EXIT_CODE
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert [item["edge_event_id"] for item in payload["cohort"]["members"]] == [
        edge_event_id
    ]
    assert len(payload["records"]) == 1
    assert payload["records"][0]["edge_event_id"] == edge_event_id
    assert payload["records"][0]["attempt_count"] == 3
    assert payload["metrics"]["cohort_total"] == 1
    assert payload["metrics"]["attributable_count"] == 1
    assert payload["cohort"]["exclusion_census"] == {}

    fresh = FalsePositiveCohortQuery(database).load()
    assert tuple(member.edge_event_id for member in fresh.members) == ()
    assert {item.reason for item in fresh.exclusions} == {"TRUE_POSITIVE"}
    with sqlite3.connect(database) as verify:
        assert verify.execute(
            "SELECT attempt_count FROM evidence_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone() == (99,)


def test_cli_analysis_opens_exactly_one_query_only_connection(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The standalone CLI must open exactly one mode=ro analysis connection."""

    database = _migrated(tmp_path)
    _seed_happy(database)
    _assert_wal(database)
    calls = _install_connect_counter(monkeypatch)

    code = main(["--edge-db", str(database)])
    captured = capsys.readouterr()

    assert code == CLEAN_EXIT_CODE
    assert captured.err == ""
    json.loads(captured.out)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs.get("uri") is True
    assert "mode=ro" in str(args[0])


def test_read_snapshot_begins_before_yield_and_closes_deterministically(
    tmp_path: Path,
) -> None:
    """The caller-owned snapshot is live before yield and closed on exit."""

    from worker.fp_attribution.snapshot import open_read_snapshot

    database = _migrated(tmp_path)
    _assert_wal(database)

    with open_read_snapshot(database) as connection:
        assert connection.isolation_level is None
        assert connection.in_transaction
        assert connection.execute("SELECT 1").fetchone() == (1,)
        held = connection

    with pytest.raises(sqlite3.ProgrammingError):
        held.execute("SELECT 1")


def test_cohort_and_evidence_reuse_caller_owned_connection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Path-bound helpers must not open a hidden second connection."""

    from worker.fp_attribution.evidence import AttributionEvidenceQuery
    from worker.fp_attribution.snapshot import open_read_snapshot

    database = _migrated(tmp_path)
    with _connect(database) as seed:
        edge_event_id = _seed_fp_event(
            seed,
            suffix="reuse",
            seqs=_complete_seqs(),
        )
        seed.commit()
    _assert_wal(database)
    calls = _install_connect_counter(monkeypatch)

    with open_read_snapshot(database) as connection:
        opened = len(calls)
        cohort = FalsePositiveCohortQuery(database).load(connection)
        evidence = AttributionEvidenceQuery(database).extract(
            connection,
            cohort=cohort,
        )

    assert opened == 1
    assert len(calls) == 1
    assert tuple(member.edge_event_id for member in cohort.members) == (edge_event_id,)
    assert tuple(record.edge_event_id for record in evidence.records) == (edge_event_id,)


def test_missing_database_never_connects_or_creates_sidecars(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """A missing --edge-db path must not open SQLite or create WAL/SHM files."""

    missing = tmp_path / "missing-edge.sqlite3"
    calls = _install_connect_counter(monkeypatch)

    code = main(["--edge-db", str(missing)])
    captured = capsys.readouterr()

    assert code == DB_UNAVAILABLE_EXIT_CODE
    assert captured.out == ""
    assert calls == []
    assert not missing.exists()
    assert not missing.with_name(f"{missing.name}-wal").exists()
    assert not missing.with_name(f"{missing.name}-shm").exists()
