"""One query-only SQLite read snapshot for standalone FP attribution."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from worker.fp_attribution.cohort import open_query_only_connection


@contextmanager
def open_read_snapshot(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open one mode=ro connection and hold an explicit read snapshot.

    The snapshot is established before the context yields so later WAL writers
    cannot change what this connection observes. The caller must keep the
    connection open through cohort, evidence, classification, and metrics.
    """

    connection = open_query_only_connection(database_path)
    try:
        connection.isolation_level = None
        connection.execute("BEGIN")
        # BEGIN is deferred, so pin the WAL snapshot with a real table read
        # before handing the connection to the analysis stages.
        connection.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()
        yield connection
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


__all__ = ["open_read_snapshot"]
