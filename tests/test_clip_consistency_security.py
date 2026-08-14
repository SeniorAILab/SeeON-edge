from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from uuid import UUID

import pytest

from worker.pipeline.output.evidence.clip_consistency_database import validate_database
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox

EVENT_ID = str(UUID(int=(4 << 76) | (2 << 62) | 301))
TIMESTAMP = "2026-08-14T00:00:00.000Z"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    state_dir = tmp_path / "state"
    clip_store = tmp_path / "clips-root"
    maintenance = tmp_path / "maintenance"
    state_dir.mkdir(mode=0o700)
    (clip_store / "clips" / ".staging").mkdir(parents=True)
    maintenance.mkdir(mode=0o700)
    database = state_dir / "worker-state.sqlite3"
    with EvidenceOutbox.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state, queued_at,
                next_attempt_at, delivery_state
            ) VALUES (?, ?, '{}', 'READY', 0, 0, 'PENDING')
            """,
            (EVENT_ID, TIMESTAMP),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state) VALUES ('clip-a', 'UNAVAILABLE')"
        )
    final = clip_store / "clips" / "clip-a"
    final.mkdir()
    (final / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "UNAVAILABLE",
                "clip_id": "clip-a",
                "camera_id": "camera-a",
                "event_refs": [EVENT_ID],
                "clip_start_at": TIMESTAMP,
                "clip_end_at": TIMESTAMP,
                "finalized_at": TIMESTAMP,
                "state_version": 2,
                "reason_code": "NO_FRAMES",
            }
        ),
        encoding="utf-8",
    )
    quiescence = maintenance / "quiescence.json"
    _write_quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    return database, clip_store, maintenance, quiescence, journal


def _write_quiescence(path: Path, database: Path, clip_store: Path) -> None:
    now = int(time.time())
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "state_db": str(database.absolute()),
                "clip_store": str(clip_store.absolute()),
                "stopped_service": "ml-worker",
                "stopped_db_writers": ["event", "config", "fault"],
                "operator_uid": os.getuid(),
                "issued_at": now - 1,
                "expires_at": now + 3599,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _apply(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    receipt = repair_clip_consistency(
        RepairRequest(
            database,
            clip_store,
            apply=True,
            maintenance_root=maintenance,
            journal_path=journal,
            quiescence_receipt=quiescence,
        )
    )
    assert receipt.backup_receipt_path is not None
    return database, clip_store, maintenance, journal, Path(receipt.backup_receipt_path)


@pytest.mark.parametrize(
    "tamper",
    (
        "unknown-key",
        "source-mode",
        "source-size",
        "source-hash",
        "source-path",
        "receipt-path",
        "backup-path",
        "duplicate-key",
        "receipt-mode",
        "backup-mode",
        "backup-bytes",
        "symlink-parent",
    ),
)
def test_backup_receipt_rejects_every_tampered_authority(
    tmp_path: Path, tamper: str
) -> None:
    database, clip_store, maintenance, journal, receipt = _apply(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if tamper == "unknown-key":
        payload["unknown"] = True
    elif tamper == "source-mode":
        payload["source_mode"] += 1
    elif tamper == "source-size":
        payload["source_size"] += 1
    elif tamper == "source-hash":
        payload["source_file_sha256"] = "0" * 64
    elif tamper == "source-path":
        payload["source_path"] = str(tmp_path / "other.sqlite3")
    elif tamper == "receipt-path":
        payload["receipt_path"] = str(tmp_path / "other.json")
    elif tamper == "backup-path":
        payload["backup_path"] = str(tmp_path / "other.sqlite3")
    elif tamper == "duplicate-key":
        raw = receipt.read_text(encoding="utf-8").removesuffix("\n").removesuffix("}")
        receipt.write_text(f'{raw},"source_size":1}}\n', encoding="utf-8")
    elif tamper == "receipt-mode":
        receipt.chmod(0o644)
    elif tamper == "backup-mode":
        Path(payload["backup_path"]).chmod(0o644)
    elif tamper == "backup-bytes":
        with Path(payload["backup_path"]).open("ab") as backup:
            backup.write(b"tampered")
    elif tamper == "symlink-parent":
        backups = receipt.parent
        outside = tmp_path / "outside-backups"
        shutil.move(backups, outside)
        backups.symlink_to(outside, target_is_directory=True)
    if tamper not in {
        "duplicate-key",
        "receipt-mode",
        "backup-mode",
        "backup-bytes",
        "symlink-parent",
    }:
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        receipt.chmod(0o600)

    with pytest.raises(ClipConsistencyError):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                resume=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=maintenance / "quiescence.json",
            )
        )


def test_expected_owner_policy_is_configurable_and_enforced(tmp_path: Path) -> None:
    database, clip_store, _, _, _ = _fixture(tmp_path)

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(database, clip_store, expected_owner_uid=os.getuid() + 1)
        )


@pytest.mark.parametrize("tamper", ("unknown", "duplicate", "mode", "path", "writers"))
def test_quiescence_receipt_is_strict(tmp_path: Path, tamper: str) -> None:
    database, clip_store, maintenance, receipt, journal = _fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if tamper == "unknown":
        payload["unknown"] = 1
    elif tamper == "duplicate":
        raw = receipt.read_text(encoding="utf-8").removesuffix("}")
        receipt.write_text(f'{raw},"operator_uid":{os.getuid()}}}', encoding="utf-8")
    elif tamper == "mode":
        receipt.chmod(0o644)
    elif tamper == "path":
        payload["state_db"] = str(tmp_path / "wrong.sqlite3")
    elif tamper == "writers":
        payload["stopped_db_writers"] = ["event", "config"]
    if tamper not in {"duplicate", "mode"}:
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        receipt.chmod(0o600)

    with pytest.raises(ClipConsistencyError, match="quiescence_invalid|unsafe_path"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=receipt,
            )
        )


@pytest.mark.parametrize("target", ("state-parent", "clip-parent", "maintenance-parent"))
def test_symlink_in_any_owned_path_parent_is_rejected(tmp_path: Path, target: str) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "state-parent":
        actual = database.parent
        moved = outside / "state"
        actual.rename(moved)
        actual.symlink_to(moved, target_is_directory=True)
    elif target == "clip-parent":
        actual = clip_store
        moved = outside / "clips"
        actual.rename(moved)
        actual.symlink_to(moved, target_is_directory=True)
    else:
        moved = outside / "maintenance"
        maintenance.rename(moved)
        maintenance.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=target == "maintenance-parent",
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
            )
        )


@pytest.mark.parametrize(
    "name,table_sql",
    (
        (
            "unique",
            """CREATE TABLE clip_events (
                clip_id TEXT NOT NULL
                    REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
                edge_event_id TEXT NOT NULL
                    REFERENCES evidence_events(edge_event_id) ON DELETE RESTRICT,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                PRIMARY KEY (clip_id, ordinal)
            ) STRICT""",
        ),
        (
            "nullability",
            """CREATE TABLE clip_events (
                clip_id TEXT REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
                edge_event_id TEXT NOT NULL UNIQUE
                    REFERENCES evidence_events(edge_event_id) ON DELETE RESTRICT,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                PRIMARY KEY (clip_id, ordinal)
            ) STRICT""",
        ),
        (
            "check",
            """CREATE TABLE clip_events (
                clip_id TEXT NOT NULL
                    REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
                edge_event_id TEXT NOT NULL UNIQUE
                    REFERENCES evidence_events(edge_event_id) ON DELETE RESTRICT,
                ordinal INTEGER NOT NULL, PRIMARY KEY (clip_id, ordinal)
            ) STRICT""",
        ),
        (
            "table-sql",
            """CREATE TABLE clip_events (
                clip_id TEXT NOT NULL
                    REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
                edge_event_id TEXT NOT NULL UNIQUE
                    REFERENCES evidence_events(edge_event_id) ON DELETE RESTRICT,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                CHECK (length(clip_id) > 0), PRIMARY KEY (clip_id, ordinal)
            ) STRICT""",
        ),
    ),
)
def test_schema9_fingerprint_rejects_constraint_and_table_sql_drift(
    tmp_path: Path, name: str, table_sql: str
) -> None:
    database, _, _, _, _ = _fixture(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE clip_events")
        connection.execute(table_sql)
    with sqlite3.connect(database) as connection:
        with pytest.raises(ClipConsistencyError, match="schema_drift"):
            validate_database(connection, now=0)


def test_verified_prebackup_can_be_reused_once_and_then_refuses_stale_source(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance, quiescence, first_journal = _fixture(tmp_path)

    def fail(stage: str) -> None:
        if stage == "apply:before_relations":
            raise RuntimeError("stop after backup")

    with pytest.raises(RuntimeError, match="stop after backup"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=True,
                maintenance_root=maintenance,
                journal_path=first_journal,
                quiescence_receipt=quiescence,
                fault_hook=fail,
            )
        )
    aborted = json.loads(first_journal.read_text(encoding="utf-8"))
    assert aborted["state"] == "ABORTED"
    backup_receipt = Path(aborted["backup_receipt_path"])

    second_journal = maintenance / "apply-2.json"
    applied = repair_clip_consistency(
        RepairRequest(
            database,
            clip_store,
            apply=True,
            maintenance_root=maintenance,
            journal_path=second_journal,
            quiescence_receipt=quiescence,
            prebackup_receipt=backup_receipt,
        )
    )
    assert applied.state == "DONE"
    assert applied.backup_receipt_path == str(backup_receipt)

    with pytest.raises(ClipConsistencyError, match="backup_receipt_stale"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=True,
                maintenance_root=maintenance,
                journal_path=maintenance / "apply-3.json",
                quiescence_receipt=quiescence,
                prebackup_receipt=backup_receipt,
            )
        )


def test_gap_free_backup_contains_committed_wal_state(tmp_path: Path) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    reader = sqlite3.connect(database, isolation_level=None)
    reader.execute("BEGIN")
    _ = reader.execute("SELECT COUNT(*) FROM evidence_events").fetchone()
    with sqlite3.connect(database) as writer:
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute(
            """INSERT INTO config_current
               (id, generation, config_version, registry_version, payload_json, saved_at)
               VALUES (1, 1, 1, 1, '{}', 1)"""
        )
    assert Path(f"{database}-wal").is_file()
    try:
        result = repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
            )
        )
    finally:
        reader.rollback()
        reader.close()
    assert result.backup_receipt_path is not None
    receipt = json.loads(Path(result.backup_receipt_path).read_text(encoding="utf-8"))
    assert receipt["source_wal_present"] is True
    with sqlite3.connect(receipt["backup_path"]) as backup:
        assert backup.execute("SELECT config_version FROM config_current").fetchone() == (1,)
