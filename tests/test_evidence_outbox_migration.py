"""Schema migration coverage for durable clip finalization state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_outbox_schema import MIGRATIONS, SCHEMA_VERSION


def test_open_migrates_v1_database_to_clip_manifest_schema(tmp_path: Path) -> None:
    # Given: a durable outbox created by the previous release.
    database = tmp_path / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    for statement in MIGRATIONS[0]:
        connection.execute(statement)
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    # When: the current worker opens the existing database.
    with EvidenceOutbox.open(database):
        pass

    # Then: schema-v2 finalization fields exist and the migration is durable.
    connection = sqlite3.connect(database)
    version = connection.execute("PRAGMA user_version").fetchone()
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(evidence_clips)").fetchall()
    }
    connection.close()
    assert version == (SCHEMA_VERSION,)
    assert {
        "local_state",
        "manifest_path",
        "state_version",
        "media_relpath",
        "sha256",
        "size_bytes",
        "unavailable_reason",
    } <= columns
