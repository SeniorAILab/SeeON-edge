"""Production-unwired DeepStream child supervision for isolated C5 dark QA."""

from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.runner import DarkRunRequest, DarkSource, run_dark_child
from worker.runtime.deepstream.source_control import SourceSnapshot, SourceState
from worker.runtime.deepstream.supervisor import (
    FATAL_CHILD_EXIT_CODE,
    DeepStreamChildSupervisor,
)

__all__ = [
    "ChildConfig",
    "DarkRunRequest",
    "DarkSource",
    "DeepStreamChildSupervisor",
    "FATAL_CHILD_EXIT_CODE",
    "SourceSnapshot",
    "SourceState",
    "run_dark_child",
]
