import asyncio
import logging
from collections.abc import Callable, Iterator

import pytest

from shared import boundary
from shared.boundary import (
    Boundary,
    ProbeFailure,
    attempt_delivery,
    cleanup_on_failure,
    degrade,
    isolate,
    probe,
    register_fatal,
    root_sink,
    translate,
)
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure


class FatalProbeError(RuntimeError):
    pass


class TypedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def fatal_registry() -> Iterator[None]:
    saved = list(boundary._fatal)
    register_fatal(FatalProbeError)
    yield
    boundary._fatal[:] = saved


def raising(error: BaseException) -> Callable[[], int]:
    def call() -> int:
        raise error

    return call


PROPAGATING = [
    FatalProbeError("accelerator lost"),
    KeyboardInterrupt(),
    SystemExit(7),
    asyncio.CancelledError(),
    GeneratorExit(),
]


@pytest.mark.parametrize("error", PROPAGATING, ids=lambda error: type(error).__name__)
def test_every_containing_helper_lets_fatal_and_control_flow_through(error: BaseException) -> None:
    with pytest.raises(type(error)), isolate(Boundary.EXPORT_ITEM, stage="item"):
        raise error
    with pytest.raises(type(error)):
        degrade(raising(error), stage="optional", default=0)
    with pytest.raises(type(error)):
        attempt_delivery(raising(error), stage="send")
    with pytest.raises(type(error)):
        probe(raising(error))
    with pytest.raises(type(error)), translate(TypedError, "typed"):
        raise error


@pytest.mark.parametrize(
    "error",
    [error for error in PROPAGATING if not isinstance(error, SystemExit)],
    ids=lambda error: type(error).__name__,
)
def test_root_sink_lets_fatal_and_interrupts_through(error: BaseException) -> None:
    with pytest.raises(type(error)):
        root_sink(raising(error), on_error_exit_code=3)


def test_register_fatal_is_idempotent() -> None:
    register_fatal(FatalProbeError, FatalProbeError)
    assert boundary._fatal.count(FatalProbeError) == 1
    assert FatalProbeError in boundary.fatal_types()


def test_isolate_contains_and_logs_stage_class_and_fields(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        with isolate(Boundary.EXPORT_ITEM, stage="export_batch", camera_id="cam-1") as outcome:
            raise ValueError("bad payload")
    assert outcome.failed
    assert outcome.error_class == "ValueError"
    [record] = caplog.records
    assert record.getMessage() == (
        "contained failure boundary=export_item stage=export_batch "
        "exception_class=ValueError camera_id=cam-1"
    )
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


def test_isolate_success_reports_no_failure(caplog: pytest.LogCaptureFixture) -> None:
    with isolate(Boundary.EXPORT_ITEM, stage="item") as outcome:
        pass
    assert not outcome.failed
    assert outcome.error_class is None
    assert caplog.records == []


def test_isolate_keeps_later_items_flowing() -> None:
    delivered: list[int] = []
    failed: list[int] = []
    for item in (1, 2, 3):
        with isolate(Boundary.EXPORT_ITEM, stage="item") as outcome:
            if item == 2:
                raise OSError("disk")
            delivered.append(item)
        if outcome.failed:
            failed.append(item)
    assert delivered == [1, 3]
    assert failed == [2]


def test_degrade_returns_value_or_default() -> None:
    assert degrade(lambda: 5, stage="optional", default=0) == 5
    assert degrade(raising(OSError("x")), stage="optional", default=0) == 0


def test_degrade_message_is_logged_verbatim_with_exc_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        result = degrade(
            raising(ConnectionError("relay down")),
            stage="status_sender",
            default=None,
            message="runtime status sender failed to start",
        )
    assert result is None
    [record] = caplog.records
    assert record.getMessage() == "runtime status sender failed to start"
    assert record.exc_info is not None


def test_degrade_without_message_logs_the_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        degrade(raising(KeyError("k")), stage="snapshot", default=None, camera_id="cam-2")
    [record] = caplog.records
    assert record.getMessage() == (
        "contained failure boundary=optional_feature stage=snapshot "
        "exception_class=KeyError camera_id=cam-2"
    )


def test_attempt_delivery_maps_failure_to_retry(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="shared.boundary"):
        failure = attempt_delivery(raising(TimeoutError("slow")), stage="heartbeat")
    assert failure == DeliveryFailure(
        DeliveryDisposition.RETRY, "UNEXPECTED", transport_error="TimeoutError: slow"
    )
    [record] = caplog.records
    assert "stage=heartbeat exception_class=TimeoutError" in record.getMessage()
    assert record.exc_info is not None


def test_attempt_delivery_passes_results_through_and_honours_code() -> None:
    assert attempt_delivery(lambda: 9, stage="send") == 9
    failure = attempt_delivery(raising(OSError("x")), stage="send", code="TRANSPORT_EXCEPTION")
    assert isinstance(failure, DeliveryFailure)
    assert failure.code == "TRANSPORT_EXCEPTION"


def test_probe_returns_reason_on_failure() -> None:
    assert probe(lambda: 1) == 1
    assert probe(raising(ImportError("no pynvml"))) == ProbeFailure("ImportError: no pynvml")


def test_cleanup_runs_and_reraises_the_same_primary() -> None:
    closed: list[str] = []
    primary = ValueError("configure failed")
    with pytest.raises(ValueError) as raised, cleanup_on_failure(lambda: closed.append("closed")):
        raise primary
    assert raised.value is primary
    assert closed == ["closed"]


def test_cleanup_does_not_run_on_success() -> None:
    closed: list[str] = []
    with cleanup_on_failure(lambda: closed.append("closed")):
        pass
    assert closed == []


def test_cleanup_failure_never_masks_the_primary_exception() -> None:
    primary = ValueError("configure failed")
    with pytest.raises(ValueError) as raised, cleanup_on_failure(raising(OSError("close failed"))):
        raise primary
    assert raised.value is primary
    assert any("close failed" in note for note in raised.value.__notes__)


def test_cleanup_runs_every_cleanup_even_after_one_fails() -> None:
    closed: list[str] = []
    with (
        pytest.raises(ValueError),
        cleanup_on_failure(raising(OSError("first")), lambda: closed.append("second")),
    ):
        raise ValueError("primary")
    assert closed == ["second"]


def test_cleanup_keeps_an_interrupt_primary_over_an_ordinary_cleanup_error() -> None:
    with pytest.raises(KeyboardInterrupt), cleanup_on_failure(raising(OSError("close"))):
        raise KeyboardInterrupt


def test_cleanup_interrupt_wins_over_an_ordinary_primary() -> None:
    primary = ValueError("primary")
    with (
        pytest.raises(KeyboardInterrupt) as raised,
        cleanup_on_failure(raising(KeyboardInterrupt())),
    ):
        raise primary
    assert raised.value.__cause__ is primary


def test_translate_wraps_with_cause_and_keeps_typed_errors() -> None:
    cause = OSError("queue unwritable")
    with (
        pytest.raises(TypedError, match="under the lock") as raised,
        translate(TypedError, "under the lock"),
    ):
        raise cause
    assert raised.value.__cause__ is cause
    already = TypedError("own")
    with pytest.raises(TypedError) as kept, translate(TypedError, "other"):
        raise already
    assert kept.value is already


def test_root_sink_maps_errors_and_exit_codes(caplog: pytest.LogCaptureFixture) -> None:
    assert root_sink(lambda: 0, on_error_exit_code=3) == 0
    assert root_sink(raising(SystemExit(5)), on_error_exit_code=3) == 5
    assert root_sink(raising(SystemExit("bye")), on_error_exit_code=3) == 3
    with caplog.at_level(logging.ERROR, logger="shared.boundary"):
        assert root_sink(raising(RuntimeError("boom")), on_error_exit_code=3, stage="cli") == 3
    [record] = caplog.records
    assert record.getMessage() == (
        "process root failed boundary=root stage=cli exception_class=RuntimeError"
    )
    assert record.exc_info is not None
