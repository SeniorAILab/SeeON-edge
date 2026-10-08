from __future__ import annotations

import threading
from collections.abc import Callable

from shared.boundary import Boundary, isolate, root_sink


def start_guarded_thread(
    name: str,
    target: Callable[[], None],
    on_death: Callable[[], None],
) -> threading.Thread:
    def body() -> int:
        target()
        return 0

    def run() -> None:
        if root_sink(body, on_error_exit_code=1, stage=f"thread:{name}") != 0:
            with isolate(Boundary.ROOT, stage=f"thread:{name}:on_death"):
                on_death()

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


__all__ = ["start_guarded_thread"]
