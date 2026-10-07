from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime.faults.record import FirstFaultRecord, persist_first_fault
from worker.runtime.state_dir import resolve_state_dir

FATAL_ACCELERATOR_EXIT_CODE: Final = 4


class _Stoppable(Protocol):
    def stop(self) -> None: ...


@final
class FaultHandler:
    def __init__(
        self,
        profile: str,
        *,
        hard_exit: Callable[[int], None] = os._exit,
        state_dir: Path | None = None,
    ) -> None:
        self._profile = profile
        self._hard_exit = hard_exit
        self._state_dir = state_dir if state_dir is not None else resolve_state_dir()
        self._loops: list[_Stoppable] = []
        self._handled = threading.Event()

    def register_loop(self, loop: _Stoppable) -> None:
        self._loops.append(loop)

    def handle(
        self,
        exc: FatalAcceleratorError,
        record: FirstFaultRecord,
    ) -> None:
        if self._handled.is_set():
            return
        self._handled.set()

        persist_first_fault(record, state_dir=self._state_dir)

        for loop in self._loops:
            with contextlib.suppress(Exception):
                loop.stop()

        self._hard_exit(FATAL_ACCELERATOR_EXIT_CODE)


__all__ = ["FATAL_ACCELERATOR_EXIT_CODE", "FaultHandler"]
