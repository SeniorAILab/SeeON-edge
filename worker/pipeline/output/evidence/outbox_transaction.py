from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from types import TracebackType
from typing import Self


class ImmediateTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> Self:
        self._connection.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self._connection.commit()
            return
        self._connection.rollback()


def write_transaction(
    connection: sqlite3.Connection,
) -> ImmediateTransaction | nullcontext[None]:
    return nullcontext() if connection.in_transaction else ImmediateTransaction(connection)


__all__ = ["ImmediateTransaction", "write_transaction"]
