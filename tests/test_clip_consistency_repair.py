from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import pytest

import worker.pipeline.output.evidence.clip_consistency_repair as repair_module
from worker.pipeline.output.evidence.clip_consistency_repair import (
    repair_clip_consistency,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import (
    ClipStoreLock,
    ClipStoreLockedError,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.manifest_models import (
    ReadyClipManifest,
    UnavailableClipManifest,
)

EVENTS = tuple(str(UUID(int=(4 << 76) | (2 << 62) | value)) for value in range(1, 7))
TIMESTAMP = "2026-08-14T00:00:00.000Z"


def _create_store(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "store"
    (store / "clips" / ".staging").mkdir(parents=True)
    database = store / "worker-state.sqlite3"
    with EvidenceOutbox.open(database):
        pass
    return store, database


def _seed(
    database: Path,
    *,
    clips: tuple[tuple[str, str], ...],
    relations: tuple[tuple[str, str, int], ...] = (),
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state, queued_at,
                next_attempt_at, delivery_state
            ) VALUES (?, ?, '{}', 'READY', 0, 0, 'PENDING')
            """,
            ((event_id, TIMESTAMP) for event_id in EVENTS),
        )
        connection.executemany(
            "INSERT INTO evidence_clips (clip_id, local_state) VALUES (?, ?)", clips
        )
        connection.executemany(
            "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
            relations,
        )


def _write_unavailable(store: Path, clip_id: str, refs: tuple[str, ...]) -> Path:
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir()
    manifest = UnavailableClipManifest(
        clip_id=clip_id,
        camera_id="camera-1",
        event_refs=refs,
        clip_start_at=TIMESTAMP,
        clip_end_at=TIMESTAMP,
        finalized_at=TIMESTAMP,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )
    path = clip_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return path


def _write_ready(store: Path, clip_id: str, refs: tuple[str, ...], ffprobe: Path) -> Path:
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir()
    media = b"\x00\x00\x00\x08moov\x00\x00\x00\x09mdatx"
    (clip_dir / "clip.mp4").write_bytes(media)
    manifest = ReadyClipManifest(
        clip_id=clip_id,
        camera_id="camera-1",
        event_refs=refs,
        clip_start_at=TIMESTAMP,
        clip_end_at=TIMESTAMP,
        finalized_at=TIMESTAMP,
        sha256=hashlib.sha256(media).hexdigest(),
        size_bytes=len(media),
        duration_ms=1000,
    )
    path = clip_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    ffprobe.write_text(
        "#!/bin/sh\nprintf '%s\\n' "
        "'[{\"not\":\"used\"}]' >/dev/null\n"
        "printf '%s\\n' "
        "'{\"streams\":[{\"codec_type\":\"video\",\"codec_name\":\"h264\","
        "\"pix_fmt\":\"yuv420p\"}],\"format\":{\"duration\":\"1.0\"}}'\n",
        encoding="utf-8",
    )
    ffprobe.chmod(0o700)
    return path


def _relations(database: Path) -> list[tuple[str, str, int]]:
    with sqlite3.connect(database) as connection:
        return [
            (str(row[0]), str(row[1]), int(row[2]))
            for row in connection.execute(
                "SELECT clip_id, edge_event_id, ordinal FROM clip_events "
                "ORDER BY clip_id, ordinal"
            )
        ]


def test_manifest_authority_repairs_relations_and_only_same_id_staging(
    tmp_path: Path,
) -> None:
    store, database = _create_store(tmp_path)
    ffprobe = tmp_path / "ffprobe"
    ready_manifest = _write_ready(store, "clip-ready", EVENTS[:2], ffprobe)
    unavailable_manifest = _write_unavailable(store, "clip-unavailable", (EVENTS[2],))
    _seed(
        database,
        clips=(
            ("clip-ready", "VERIFIED"),
            ("clip-unavailable", "UNAVAILABLE"),
            ("clip-corrupt", "CORRUPT"),
            ("clip-awaiting", "AWAITING_FINALIZE"),
        ),
        relations=(
            ("clip-ready", EVENTS[1], 0),
            ("clip-ready", EVENTS[0], 1),
            ("clip-ready", EVENTS[3], 2),
            ("clip-awaiting", EVENTS[2], 0),
            ("clip-awaiting", EVENTS[5], 1),
            ("clip-corrupt", EVENTS[4], 0),
        ),
    )
    overlap = store / "clips" / ".staging" / "clip-ready"
    overlap.mkdir()
    (overlap / "partial.bin").write_bytes(b"stale")
    unrelated = store / "clips" / ".staging" / "unrelated"
    unrelated.mkdir()
    ready_before = (ready_manifest.read_bytes(), (store / "clips/clip-ready/clip.mp4").read_bytes())
    unavailable_before = unavailable_manifest.read_bytes()

    dry = repair_clip_consistency(RepairRequest(store, ffprobe_bin=str(ffprobe)))

    assert dry.mode == "dry-run"
    assert dry.counters.ready_finals == 1
    assert dry.counters.unavailable_finals == 1
    assert dry.counters.relations_before == 6
    assert dry.counters.relations_after == 5
    assert dry.counters.relations_deleted == 4
    assert dry.counters.relations_inserted == 3
    assert dry.counters.staging_to_delete == 1
    assert overlap.exists()

    applied = repair_clip_consistency(
        RepairRequest(store, apply=True, ffprobe_bin=str(ffprobe))
    )

    assert applied.mode == "apply"
    assert applied.counters.staging_deleted == 1
    assert applied.counters.staging_to_delete == 0
    assert _relations(database) == [
        ("clip-awaiting", EVENTS[5], 1),
        ("clip-corrupt", EVENTS[4], 0),
        ("clip-ready", EVENTS[0], 0),
        ("clip-ready", EVENTS[1], 1),
        ("clip-unavailable", EVENTS[2], 0),
    ]
    assert not overlap.exists()
    assert unrelated.is_dir()
    ready_after = (
        ready_manifest.read_bytes(),
        (store / "clips/clip-ready/clip.mp4").read_bytes(),
    )
    assert ready_after == ready_before
    assert unavailable_manifest.read_bytes() == unavailable_before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0] == 6
        assert connection.execute("SELECT COUNT(*) FROM evidence_clips").fetchone()[0] == 4
    assert applied.backup_receipt_path is not None
    backup_receipt = json.loads(Path(applied.backup_receipt_path).read_text(encoding="utf-8"))
    backup = Path(backup_receipt["backup_path"])
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_receipt["backup_sha256"]
    assert applied.receipt_path is not None and Path(applied.receipt_path).is_file()

    second = repair_clip_consistency(RepairRequest(store, ffprobe_bin=str(ffprobe)))
    assert second.counters.changes == 0
    assert second.counters.relations_before == second.counters.relations_after == 5


def test_same_refs_with_wrong_ordinals_are_rewritten(tmp_path: Path) -> None:
    store, database = _create_store(tmp_path)
    _write_unavailable(store, "clip-a", EVENTS[:2])
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[0], 3), ("clip-a", EVENTS[1], 4)),
    )

    dry = repair_clip_consistency(RepairRequest(store))

    assert dry.counters.relations_deleted == 2
    assert dry.counters.relations_inserted == 2
    repair_clip_consistency(RepairRequest(store, apply=True))
    assert _relations(database) == [
        ("clip-a", EVENTS[0], 0),
        ("clip-a", EVENTS[1], 1),
    ]


def test_apply_rolls_back_database_and_staging_on_transaction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, database = _create_store(tmp_path)
    _write_unavailable(store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = store / "clips/.staging/clip-a"
    staging.mkdir()
    before = _relations(database)

    def fail_verification(*_args: object) -> None:
        raise ClipConsistencyError("injected", "transaction failure")

    monkeypatch.setattr(repair_module, "_verify_applied", fail_verification)

    with pytest.raises(ClipConsistencyError, match="injected"):
        repair_clip_consistency(RepairRequest(store, apply=True))

    assert staging.is_dir()
    assert _relations(database) == before


def test_apply_rolls_back_staging_moves_and_relations_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, database = _create_store(tmp_path)
    _write_unavailable(store, "clip-a", (EVENTS[0],))
    _write_unavailable(store, "clip-b", (EVENTS[1],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"), ("clip-b", "UNAVAILABLE")),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    first = store / "clips/.staging/clip-a"
    second = store / "clips/.staging/clip-b"
    first.mkdir()
    second.mkdir()
    before = _relations(database)
    real_replace = repair_module.os.replace
    calls = 0

    def fail_second_move(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected move failure")
        real_replace(source, destination)

    monkeypatch.setattr(repair_module.os, "replace", fail_second_move)

    with pytest.raises(OSError, match="injected move failure"):
        repair_clip_consistency(RepairRequest(store, apply=True))

    assert first.is_dir() and second.is_dir()
    assert _relations(database) == before

    existing_receipt = next(
        (store / "maintenance-backups").glob("*.receipt.json")
    )
    monkeypatch.setattr(repair_module.os, "replace", real_replace)
    applied = repair_clip_consistency(
        RepairRequest(store, apply=True, prebackup_receipt=existing_receipt)
    )
    assert applied.backup_receipt_path == str(existing_receipt.resolve())
    assert not first.exists() and not second.exists()


def test_refuses_active_lease_and_worker_store_lock(tmp_path: Path) -> None:
    store, database = _create_store(tmp_path)
    _seed(database, clips=(("clip-awaiting", "AWAITING_FINALIZE"),))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET state='IN_FLIGHT', lease_owner='sender', "
            "lease_expires_at=? WHERE edge_event_id=?",
            (time.time() + 3600, EVENTS[0]),
        )
    with pytest.raises(ClipConsistencyError, match="active_lease"):
        repair_clip_consistency(RepairRequest(store))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET state='READY', lease_owner=NULL, lease_expires_at=NULL"
        )
    with ClipStoreLock.acquire(store):
        with pytest.raises(ClipStoreLockedError):
            repair_clip_consistency(RepairRequest(store))


@pytest.mark.parametrize("fault", ("malformed", "corrupt-ready", "symlink"))
def test_refuses_malformed_corrupt_or_symlinked_finals(
    tmp_path: Path, fault: str
) -> None:
    store, database = _create_store(tmp_path)
    _seed(database, clips=(("clip-bad", "CORRUPT"),))
    clip_dir = store / "clips/clip-bad"
    clip_dir.mkdir()
    if fault == "malformed":
        (clip_dir / "manifest.json").write_text("{broken", encoding="utf-8")
    elif fault == "corrupt-ready":
        ffprobe = tmp_path / "ffprobe"
        _write_ready_in_existing(clip_dir, EVENTS[0], ffprobe)
        (clip_dir / "clip.mp4").write_bytes(b"changed")
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (clip_dir / "manifest.json").symlink_to(outside)
    with pytest.raises(ClipConsistencyError, match="final_invalid|unsafe_path"):
        repair_clip_consistency(
            RepairRequest(store, ffprobe_bin=str(tmp_path / "ffprobe"))
        )


def _write_ready_in_existing(clip_dir: Path, event_id: str, ffprobe: Path) -> None:
    clip_dir.rmdir()
    _write_ready(clip_dir.parents[1], clip_dir.name, (event_id,), ffprobe)


def test_command_defaults_to_dry_run_and_prints_only_receipt_data(tmp_path: Path) -> None:
    store, database = _create_store(tmp_path)
    _write_unavailable(store, "clip-a", (EVENTS[0],))
    _seed(database, clips=(("clip-a", "UNAVAILABLE"),))
    repository = Path(__file__).parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/repair_clip_consistency.py", str(store)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["changes"] == 1
    assert completed.stderr == ""
    assert _relations(database) == []
    assert "manifest.json" not in completed.stdout


def test_refuses_schema_foreign_key_drift_and_stale_backup_receipt(tmp_path: Path) -> None:
    store, database = _create_store(tmp_path)
    _write_unavailable(store, "clip-a", (EVENTS[0],))
    _seed(database, clips=(("clip-a", "UNAVAILABLE"),))
    first = repair_clip_consistency(RepairRequest(store, apply=True))
    assert first.backup_receipt_path is not None
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 8")
    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(RepairRequest(store))
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 9")
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "INSERT INTO clip_events VALUES ('clip-a', 'missing-event', 9)"
        )
    with pytest.raises(ClipConsistencyError, match="foreign_key_drift"):
        repair_clip_consistency(RepairRequest(store))
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM clip_events WHERE edge_event_id='missing-event'")
        connection.execute("UPDATE evidence_events SET queued_at = 1")
    with pytest.raises(ClipConsistencyError, match="backup_stale"):
        repair_clip_consistency(
            RepairRequest(
                store,
                apply=True,
                prebackup_receipt=Path(first.backup_receipt_path),
            )
        )
