from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from worker.pipeline.output.evidence.clip_consistency_authority import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_database import validate_database
from worker.pipeline.output.evidence.clip_consistency_operation import (
    image_artifact_identity,
)
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    RepairRequest as _RepairRequest,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox

EVENT_ID = str(UUID(int=(4 << 76) | (2 << 62) | 101))
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


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    state_dir = tmp_path / "worker-state"
    clip_store = tmp_path / "clip-store"
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
    manifest = {
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
        "legacy_extra": {"allowed": True},
    }
    final = clip_store / "clips" / "clip-a"
    final.mkdir()
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return database, clip_store, maintenance


def _command(
    database: Path,
    clip_store: Path,
    *extra: str,
    with_authority: bool = True,
) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/repair_clip_consistency.py",
            "--state-db",
            str(database),
            "--clip-store",
            str(clip_store),
            *(_authority_args(database, clip_store) if with_authority else ()),
            *extra,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def _quiescence(path: Path, database: Path, clip_store: Path) -> None:
    authority = _authority(database, clip_store)
    payload = {
        "format_version": 3,
        "state_db": str(database.resolve()),
        "clip_store": str(clip_store.resolve()),
        "stopped_service": "ml-worker",
        "stopped_db_writers": ["event", "config", "fault"],
        "operator_uid": authority.state_uid,
        "authority_sha256": authority.sha256,
        "operation_digest_version": 1,
        "operation_digest": "0" * 64,
        "image_artifact_identity": image_artifact_identity(authority),
        **authority.to_dict(),
        "issued_at": int(time.time()) - 1,
        "expires_at": int(time.time()) + 3599,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def _authority_args(database: Path, clip_store: Path) -> tuple[str, ...]:
    return (
        "--state-uid",
        str(database.stat().st_uid),
        "--state-gid",
        str(database.stat().st_gid),
        "--state-db-mode",
        f"{stat.S_IMODE(database.stat().st_mode):04o}",
        "--state-dir-mode",
        f"{stat.S_IMODE(database.parent.stat().st_mode):04o}",
        "--clip-uid",
        str(clip_store.stat().st_uid),
        "--clip-gid",
        str(clip_store.stat().st_gid),
        "--clip-dir-mode",
        f"{stat.S_IMODE(clip_store.stat().st_mode):04o}",
        "--tool-revision",
        TOOL_REVISION,
    )


@pytest.mark.parametrize("identifier", (-1, 2**32 - 1, 2**32, 10**100))
def test_cli_rejects_unsupported_linux_uid_range(
    tmp_path: Path, identifier: int
) -> None:
    database, clip_store, _ = _layout(tmp_path)

    completed = _command(database, clip_store, "--state-uid", str(identifier))

    assert completed.returncode == 2
    assert json.loads(completed.stderr)["error"].startswith("authority_invalid:")


def test_cli_requires_explicit_split_authority(tmp_path: Path) -> None:
    database, clip_store, _ = _layout(tmp_path)

    missing = _command(database, clip_store, with_authority=False)
    completed = _command(database, clip_store)

    assert missing.returncode == 2
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["mode"] == "dry-run"


def test_apply_refuses_concurrent_ordinary_database_writer_without_mutation(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)
    writer_ready = threading.Event()
    release_writer = threading.Event()

    def writer() -> None:
        with sqlite3.connect(database, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE config_current SET saved_at = saved_at")
            writer_ready.set()
            assert release_writer.wait(timeout=3)
            connection.rollback()

    thread = threading.Thread(target=writer)
    thread.start()
    assert writer_ready.wait(timeout=3)
    try:
        completed = _command(
            database,
            clip_store,
            "--apply",
            "--maintenance-root",
            str(maintenance),
            "--journal",
            str(journal),
            "--quiescence-receipt",
            str(quiescence),
        )
    finally:
        release_writer.set()
        thread.join(timeout=3)

    assert completed.returncode == 2
    assert "database_busy" in completed.stderr
    assert not journal.exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM clip_events").fetchone()[0] == 0


def test_ordinary_writer_is_excluded_after_tool_holds_write_boundary(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)
    tool_holds_boundary = threading.Event()
    release_tool = threading.Event()
    result: list[object] = []

    def pause_after_boundary(stage: str) -> None:
        if stage == "backup:file_fsynced":
            tool_holds_boundary.set()
            assert release_tool.wait(timeout=3)

    def run_tool() -> None:
        result.append(
            repair_clip_consistency(
                RepairRequest(
                    database,
                    clip_store,
                    apply=True,
                    maintenance_root=maintenance,
                    journal_path=journal,
                    quiescence_receipt=quiescence,
                    fault_hook=pause_after_boundary,
                )
            )
        )

    thread = threading.Thread(target=run_tool)
    thread.start()
    assert tool_holds_boundary.wait(timeout=3)
    try:
        with sqlite3.connect(database, isolation_level=None, timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("BEGIN IMMEDIATE")
    finally:
        release_tool.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert len(result) == 1
    assert journal.is_file()


@pytest.mark.parametrize("duplicate_key", ("event_refs", "clip_id", "sha256"))
def test_authority_manifest_rejects_duplicate_keys(
    tmp_path: Path, duplicate_key: str
) -> None:
    database, clip_store, _ = _layout(tmp_path)
    manifest = clip_store / "clips" / "clip-a" / "manifest.json"
    payload = manifest.read_text(encoding="utf-8").removesuffix("}")
    duplicate_value = {
        "event_refs": json.dumps([EVENT_ID]),
        "clip_id": json.dumps("clip-a"),
        "sha256": json.dumps("0" * 64),
    }[duplicate_key]
    prefix = f'{payload},"sha256":{json.dumps("1" * 64)}' if duplicate_key == "sha256" else payload
    manifest.write_text(
        f'{prefix},"{duplicate_key}":{duplicate_value}}}', encoding="utf-8"
    )

    completed = _command(database, clip_store)

    assert completed.returncode == 2
    assert "final_invalid" in completed.stderr


def test_authority_manifest_rejects_nested_duplicate_keys(tmp_path: Path) -> None:
    database, clip_store, _ = _layout(tmp_path)
    manifest = clip_store / "clips" / "clip-a" / "manifest.json"
    payload = manifest.read_text(encoding="utf-8").removesuffix("}")
    manifest.write_text(
        f'{payload},"nested_legacy":{{"identity":1,"identity":2}}}}',
        encoding="utf-8",
    )

    completed = _command(database, clip_store)

    assert completed.returncode == 2
    assert "final_invalid" in completed.stderr


@pytest.mark.parametrize(
    "drift_sql",
    (
        "DROP INDEX evidence_events_claim_idx",
        "DROP INDEX evidence_clips_publish_idx",
        "DROP INDEX evidence_events_delivery_idx",
    ),
)
def test_schema9_fingerprint_rejects_required_index_drift(
    tmp_path: Path, drift_sql: str
) -> None:
    database, _, _ = _layout(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(drift_sql)
    with sqlite3.connect(database) as connection:
        with pytest.raises(ClipConsistencyError, match="schema_drift"):
            validate_database(connection, now=0)


def test_apply_requires_recoverable_prepared_journal(tmp_path: Path) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)

    completed = _command(
        database,
        clip_store,
        "--apply",
        "--maintenance-root",
        str(maintenance),
        "--journal",
        str(journal),
        "--quiescence-receipt",
        str(quiescence),
    )

    assert completed.returncode == 0
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert payload["state"] == "DONE"
    assert payload["source_state_sha256"]
    assert payload["plan_sha256"]
    assert payload["backup_receipt_path"]
