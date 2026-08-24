"""Dedicated isolated Python PID-1 dark runner; normal WorkerRuntime never imports it."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from types import FrameType
from typing import Final

from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.fault import persist_first_fault
from worker.runtime.deepstream.supervisor import (
    FATAL_CHILD_EXIT_CODE,
    ChildFatalError,
    ChildStartupError,
    DeepStreamChildSupervisor,
)

CLEAN_EXIT_CODE: Final = 0


@dataclass(frozen=True, slots=True)
class DarkSource:
    camera_id: str
    uri: str


@dataclass(frozen=True, slots=True)
class DarkRunRequest:
    child: ChildConfig
    sources: tuple[DarkSource, ...] = ()
    inject_fatal: str | None = None


def run_dark_child(request: DarkRunRequest) -> int:
    """Run one child to terminal exit; any non-graceful loss maps to container exit 4."""
    supervisor = DeepStreamChildSupervisor(request.child)
    previous_handlers = [
        (signum, signal.getsignal(signum)) for signum in (signal.SIGINT, signal.SIGTERM)
    ]

    def stop_handler(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        supervisor.stop()

    try:
        for signum, _handler in previous_handlers:
            _ = signal.signal(signum, stop_handler)
        supervisor.start()
        for source in request.sources:
            _ = supervisor.sources.add(source.camera_id, source.uri)
        if request.inject_fatal is not None:
            supervisor.fatal(request.inject_fatal)
        try:
            return supervisor.wait()
        except ChildFatalError:
            return FATAL_CHILD_EXIT_CODE
    except ChildStartupError as error:
        _ = persist_first_fault(
            request.child.first_fault_path,
            category=error.code,
            exit_code=FATAL_CHILD_EXIT_CODE,
            worker_boot_id=request.child.worker_boot_id,
            child_instance_id=request.child.child_instance_id,
        )
        return FATAL_CHILD_EXIT_CODE
    finally:
        supervisor.stop()
        for signum, handler in previous_handlers:
            _ = signal.signal(signum, handler)


__all__ = [
    "CLEAN_EXIT_CODE",
    "DarkRunRequest",
    "DarkSource",
    "run_dark_child",
]
