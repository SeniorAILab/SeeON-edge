from __future__ import annotations

import fcntl
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from backend.app.edge_db.compact_schema import SCHEMA_19_STATEMENTS
from backend.app.edge_db.migration.compatibility import SCHEMA_19_IDENTITY
from backend.app.edge_db.migration.sqlite_functions import register_edge_db_functions
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION


def create_schema19_source(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA journal_mode = WAL").fetchone() != ("wal",):
            raise RuntimeError("source database could not enter WAL mode")
        connection.execute("PRAGMA synchronous = FULL")
        register_edge_db_functions(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA_19_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum, applied_at)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                SCHEMA_19_IDENTITY,
            )
            connection.execute(f"PRAGMA user_version = {EDGE_DATABASE_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return path


def open_source_writer(source: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(source, isolation_level=None)
    register_edge_db_functions(connection)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA wal_autocheckpoint = 0")
    return connection


@contextmanager
def hold_runtime_lock(source: Path) -> Iterator[None]:
    descriptor = os.open(source.parent / "deployment.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


__all__ = ["create_schema19_source", "hold_runtime_lock", "open_source_writer"]
