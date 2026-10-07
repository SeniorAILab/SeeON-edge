import logging
import signal
from collections.abc import Callable

import pytest

from worker import __main__ as worker_main


class Runtime:
    def __init__(self, run: Callable[[], None]) -> None:
        self.run = run
        self.stopped = 0

    def stop(self) -> None:
        self.stopped += 1


def raising(error: BaseException) -> Callable[[], None]:
    def run() -> None:
        raise error

    return run


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (lambda: None, worker_main.CLEAN_SHUTDOWN_EXIT_CODE),
        (raising(SystemExit(3)), 3),
        (raising(SystemExit(0)), 0),
        (raising(SystemExit(None)), worker_main.GENERIC_RUNTIME_ERROR_EXIT_CODE),
        (raising(SystemExit("bye")), worker_main.GENERIC_RUNTIME_ERROR_EXIT_CODE),
        (raising(RuntimeError("boom")), worker_main.GENERIC_RUNTIME_ERROR_EXIT_CODE),
    ],
    ids=["clean", "exit-3", "exit-0", "exit-none", "exit-str", "error"],
)
def test_run_runtime_keeps_every_exit_code(run: Callable[[], None], expected: int) -> None:
    assert worker_main._run_runtime(Runtime(run)) == expected


def test_run_runtime_logs_an_unexpected_failure_through_root_sink(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="shared.boundary"):
        worker_main._run_runtime(Runtime(raising(RuntimeError("boom"))))
    [record] = caplog.records
    assert record.getMessage() == (
        "process root failed boundary=root stage=worker_runtime exception_class=RuntimeError"
    )
    assert record.exc_info is not None


def test_run_runtime_lets_an_interrupt_through_and_restores_signal_handlers() -> None:
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    with pytest.raises(KeyboardInterrupt):
        worker_main._run_runtime(Runtime(raising(KeyboardInterrupt())))
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_run_runtime_installs_a_signal_handler_that_stops_the_runtime() -> None:
    seen: list[object] = []

    def run() -> None:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        seen.append(handler)

    runtime = Runtime(run)
    assert worker_main._run_runtime(runtime) == worker_main.CLEAN_SHUTDOWN_EXIT_CODE
    assert runtime.stopped == 1
    assert seen
