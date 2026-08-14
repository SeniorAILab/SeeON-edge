from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from worker.pipeline.output.evidence.clip_consistency_authority import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_backup import (
    verify_backup_receipt_for_resume,
)
from worker.pipeline.output.evidence.clip_consistency_database import validate_database
from worker.pipeline.output.evidence.clip_consistency_io import validate_under_root
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    RepairRequest as _RepairRequest,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox

EVENT_ID = str(UUID(int=(4 << 76) | (2 << 62) | 301))
TIMESTAMP = "2026-08-14T00:00:00.000Z"
TOOL_REVISION = "31de1430758d05d744686be6098e00641f4ea4d9"


def _authority(database: Path, clip_store: Path) -> RepairAuthority:
    return RepairAuthority(
        state_uid=database.stat().st_uid,
        state_gid=database.stat().st_gid,
        state_db_mode=stat.S_IMODE(database.stat().st_mode),
        state_dir_mode=stat.S_IMODE(database.parent.stat().st_mode),
        clip_uid=clip_store.stat().st_uid,
        clip_gid=clip_store.stat().st_gid,
        clip_dir_mode=stat.S_IMODE(clip_store.stat().st_mode),
        tool_revision=TOOL_REVISION,
    )


def RepairRequest(
    state_db: Path, clip_store: Path, *args: Any, **kwargs: Any
) -> _RepairRequest:
    kwargs.setdefault("authority", _authority(state_db, clip_store))
    return _RepairRequest(state_db, clip_store, *args, **kwargs)


def _split_clip_gid(clip_store: Path) -> int:
    alternatives = [gid for gid in os.getgroups() if gid != os.getgid()]
    if not alternatives:
        pytest.skip("test user has no supplementary group for split-GID ownership")
    gid = alternatives[0]
    for root, directories, files in os.walk(clip_store):
        os.chown(root, -1, gid)
        for name in (*directories, *files):
            os.chown(Path(root) / name, -1, gid)
    for path in (clip_store, clip_store / "clips", clip_store / "clips/.staging"):
        path.chmod(0o775)
    return gid


def _receipt_binding(receipt: Path) -> tuple[RepairAuthority, Path]:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    authority = RepairAuthority(
        **{key: payload[key] for key in RepairAuthority.__dataclass_fields__}
    )
    return authority, Path(payload["clip_store"])


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
    authority = _authority(database, clip_store)
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "state_db": str(database.absolute()),
                "clip_store": str(clip_store.absolute()),
                "stopped_service": "ml-worker",
                "stopped_db_writers": ["event", "config", "fault"],
                "operator_uid": authority.state_uid,
                "authority_sha256": authority.sha256,
                **authority.to_dict(),
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
        "state-gid",
        "clip-gid",
        "tool-revision",
        "clip-store",
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
    elif tamper == "state-gid":
        payload["state_gid"] += 1
    elif tamper == "clip-gid":
        payload["clip_gid"] += 1
    elif tamper == "tool-revision":
        payload["tool_revision"] = "0" * 40
    elif tamper == "clip-store":
        payload["clip_store"] = str(tmp_path / "wrong-store")
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


@pytest.mark.parametrize(
    "field,value",
    (
        ("state_gid", 99999),
        ("clip_gid", 99999),
        ("clip_dir_mode", 0o777),
        ("tool_revision", "0" * 40),
    ),
)
def test_resume_rejects_tampered_journal_authority(
    tmp_path: Path, field: str, value: int | str
) -> None:
    database, clip_store, maintenance, journal, _ = _apply(tmp_path)
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload[field] = value
    journal.write_text(json.dumps(payload), encoding="utf-8")
    journal.chmod(0o600)

    with pytest.raises(ClipConsistencyError, match="journal_invalid"):
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


def test_legitimate_split_authority_dry_run_apply_and_resume(tmp_path: Path) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    clip_gid = _split_clip_gid(clip_store)
    _write_quiescence(quiescence, database, clip_store)
    authority = _authority(database, clip_store)
    assert authority.state_gid != clip_gid == authority.clip_gid

    dry = repair_clip_consistency(
        RepairRequest(database, clip_store, authority=authority)
    )
    fired = False

    def interrupt_after_commit(stage: str) -> None:
        nonlocal fired
        if stage == "apply:after_commit" and not fired:
            fired = True
            raise RuntimeError("resume split authority")

    with pytest.raises(RuntimeError, match="resume split authority"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                authority=authority,
                apply=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
                fault_hook=interrupt_after_commit,
            )
        )
    resumed = repair_clip_consistency(
        RepairRequest(
            database,
            clip_store,
            authority=authority,
            resume=True,
            maintenance_root=maintenance,
            journal_path=journal,
            quiescence_receipt=quiescence,
        )
    )

    assert dry.state == "DRY_RUN"
    assert resumed.state == "DONE"


@pytest.mark.parametrize("entry", ("final", "staging"))
def test_split_authority_rejects_mixed_clip_entry_owner(
    tmp_path: Path, entry: str
) -> None:
    database, clip_store, _, _, _ = _fixture(tmp_path)
    _split_clip_gid(clip_store)
    target = (
        clip_store / "clips/clip-a/manifest.json"
        if entry == "final"
        else clip_store / "clips/.staging"
    )
    os.chown(target, -1, os.getgid())

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(database, clip_store, authority=_authority(database, clip_store))
        )


def test_prepared_resume_rejects_clip_owner_change(tmp_path: Path) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    _split_clip_gid(clip_store)
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    os.chown(staging, -1, clip_store.stat().st_gid)
    _write_quiescence(quiescence, database, clip_store)
    authority = _authority(database, clip_store)

    def interrupt_after_commit(stage: str) -> None:
        if stage == "apply:after_commit":
            raise RuntimeError("prepared after commit")

    with pytest.raises(RuntimeError, match="prepared after commit"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                authority=authority,
                apply=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
                fault_hook=interrupt_after_commit,
            )
        )
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "PREPARED"
    os.chown(clip_store / "clips/clip-a/manifest.json", -1, authority.state_gid)

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                authority=authority,
                resume=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
            )
        )


def test_writable_authority_ancestor_is_rejected(tmp_path: Path) -> None:
    database, clip_store, _, _, _ = _fixture(tmp_path)
    tmp_path.chmod(0o777)

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(database, clip_store, authority=_authority(database, clip_store))
        )


def test_split_owner_policy_is_explicit_and_enforced(tmp_path: Path) -> None:
    database, clip_store, _, _, _ = _fixture(tmp_path)
    wrong = replace(_authority(database, clip_store), clip_uid=os.getuid() + 1)

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(
            RepairRequest(database, clip_store, authority=wrong)
        )


@pytest.mark.parametrize(
    "tamper", ("unknown", "duplicate", "mode", "path", "writers", "authority")
)
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
    elif tamper == "authority":
        payload["clip_gid"] += 1
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


@pytest.mark.parametrize("object_kind", ("trigger", "view", "table"))
def test_schema9_rejects_every_unexpected_sqlite_master_object_before_plan(
    tmp_path: Path,
    object_kind: str,
) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    statements = {
        "trigger": """
            CREATE TRIGGER mutate_event_after_relation
            AFTER INSERT ON clip_events BEGIN
                UPDATE evidence_events SET payload_json = '{\"mutated\":true}';
            END
        """,
        "view": "CREATE VIEW leaked_events AS SELECT * FROM evidence_events",
        "table": "CREATE TABLE unexpected_authority (value TEXT) STRICT",
    }
    with sqlite3.connect(database) as connection:
        before = connection.execute(
            "SELECT payload_json FROM evidence_events WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone()
        connection.execute(statements[object_kind])

    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(
            RepairRequest(
                database,
                clip_store,
                apply=True,
                maintenance_root=maintenance,
                journal_path=journal,
                quiescence_receipt=quiescence,
            )
        )

    assert not journal.exists()
    assert not (maintenance / "backups").exists()
    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT payload_json FROM evidence_events WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone()
        assert connection.execute("SELECT COUNT(*) FROM clip_events").fetchone() == (0,)
    assert after == before


@pytest.mark.parametrize(
    "drop_sql",
    (
        "DROP TABLE faults",
        "DROP TABLE config_history",
        "DROP INDEX evidence_events_claim_idx",
    ),
)
def test_schema9_rejects_every_missing_canonical_object(
    tmp_path: Path,
    drop_sql: str,
) -> None:
    database, _, _, _, _ = _fixture(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(drop_sql)
    with sqlite3.connect(database) as connection:
        with pytest.raises(ClipConsistencyError, match="schema_drift"):
            validate_database(connection, now=0)


@pytest.mark.parametrize("authority", ("state", "clip", "maintenance", "journal", "proof"))
def test_every_request_authority_rejects_lexical_parent_components(
    tmp_path: Path,
    authority: str,
) -> None:
    database, clip_store, maintenance, quiescence, journal = _fixture(tmp_path)
    (database.parent / "nested").mkdir()
    (clip_store / "nested").mkdir()
    (maintenance / "nested").mkdir()
    aliases = {
        "state": database.parent / "nested" / ".." / database.name,
        "clip": clip_store / "nested" / "..",
        "maintenance": maintenance / "nested" / "..",
        "journal": maintenance / "nested" / ".." / journal.name,
        "proof": maintenance / "nested" / ".." / quiescence.name,
    }
    request = RepairRequest(
        aliases["state"] if authority == "state" else database,
        aliases["clip"] if authority == "clip" else clip_store,
        apply=authority not in {"state", "clip"},
        maintenance_root=(
            aliases["maintenance"] if authority == "maintenance" else maintenance
        ),
        journal_path=aliases["journal"] if authority == "journal" else journal,
        quiescence_receipt=(
            aliases["proof"] if authority == "proof" else quiescence
        ),
    )

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        repair_clip_consistency(request)


def test_root_containment_rejects_lexical_and_resolved_escape_forms(
    tmp_path: Path,
) -> None:
    root = tmp_path / "maintenance"
    nested = root / "nested"
    escape = tmp_path / "escape"
    nested.mkdir(parents=True)
    escape.mkdir()
    candidates = (
        root / ".." / "escape",
        nested / "." / ".." / ".." / "escape",
        Path(str(root) + "/nested/../../escape"),
    )
    for candidate in candidates:
        with pytest.raises(ClipConsistencyError, match="unsafe_path"):
            validate_under_root(candidate, root, allow_missing_leaf=False)


@pytest.mark.parametrize("field", ("source_size", "source_file_sha256"))
def test_reusable_backup_receipt_rejects_tampered_raw_source_facts(
    tmp_path: Path,
    field: str,
) -> None:
    _, _, maintenance, _, receipt = _apply(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = payload[field] + 1 if field == "source_size" else "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    receipt.chmod(0o600)
    authority, clip_store = _receipt_binding(receipt)

    with pytest.raises(ClipConsistencyError, match="backup_receipt_invalid"):
        verify_backup_receipt_for_resume(
            receipt,
            maintenance_root=maintenance,
            clip_store=clip_store,
            authority=authority,
        )


def test_backup_authority_path_rejects_lexical_parent_alias(tmp_path: Path) -> None:
    _, _, maintenance, _, receipt = _apply(tmp_path)
    receipt_alias = receipt.parent / "nested"
    receipt_alias.mkdir()
    lexical = receipt_alias / ".." / receipt.name
    authority, clip_store = _receipt_binding(receipt)

    with pytest.raises(ClipConsistencyError, match="unsafe_path"):
        verify_backup_receipt_for_resume(
            lexical,
            maintenance_root=maintenance,
            clip_store=clip_store,
            authority=authority,
        )


@pytest.mark.parametrize(
    "field",
    ("source_path", "source_wal_path", "backup_path", "receipt_path"),
)
def test_backup_receipt_rejects_lexical_parent_in_every_advertised_path(
    tmp_path: Path,
    field: str,
) -> None:
    _, _, maintenance, _, receipt = _apply(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    original = Path(payload[field])
    nested = original.parent / "nested-authority"
    nested.mkdir(exist_ok=True)
    payload[field] = str(nested / ".." / original.name)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    receipt.chmod(0o600)
    authority, clip_store = _receipt_binding(receipt)

    with pytest.raises(ClipConsistencyError, match="unsafe_path|backup_receipt_invalid"):
        verify_backup_receipt_for_resume(
            receipt,
            maintenance_root=maintenance,
            clip_store=clip_store,
            authority=authority,
        )


def test_edge_worker_ci_smokes_packaged_maintenance_cli() -> None:
    workflow = Path(".github/workflows/edge-worker-image.yml").read_text(encoding="utf-8")

    assert "scripts/repair_clip_consistency.py --help" in workflow


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
