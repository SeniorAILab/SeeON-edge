from __future__ import annotations

import sqlite3
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox_schema import MIGRATIONS, SCHEMA_VERSION
from worker.pipeline.output.evidence.evidence_outbox_types import (
    DatabaseSettings,
    NewerSchemaVersionError,
)


def open_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    version_row = connection.execute("PRAGMA user_version").fetchone()
    version = 0 if version_row is None else int(version_row[0])
    if version > SCHEMA_VERSION:
        connection.close()
        raise NewerSchemaVersionError(found=version, supported=SCHEMA_VERSION)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA journal_mode = WAL")
        _migrate(connection, version)
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def database_settings(connection: sqlite3.Connection) -> DatabaseSettings:
    journal = connection.execute("PRAGMA journal_mode").fetchone()
    synchronous = connection.execute("PRAGMA synchronous").fetchone()
    foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
    return DatabaseSettings(
        journal_mode="" if journal is None else str(journal[0]),
        synchronous=0 if synchronous is None else int(synchronous[0]),
        foreign_keys=0 if foreign_keys is None else int(foreign_keys[0]),
    )


def _migrate(connection: sqlite3.Connection, version: int) -> None:
    if version == SCHEMA_VERSION:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            for statement in MIGRATIONS[target_version - 1]:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {target_version}")
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise


__all__ = ["database_settings", "open_connection"]
