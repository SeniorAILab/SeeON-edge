from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionRecordSink(Protocol):
    def try_emit(self, record: object) -> bool: ...


__all__ = ["ExecutionRecordSink"]
