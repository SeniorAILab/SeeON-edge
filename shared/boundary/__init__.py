import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

LOGGER: Final = logging.getLogger("shared.boundary")

ALWAYS_PROPAGATE: Final[tuple[type[BaseException], ...]] = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    asyncio.CancelledError,
)

_fatal: list[type[BaseException]] = []
_translation_targets: list[type[Exception]] = []


class Boundary(StrEnum):
    EXPORT_ITEM = "export_item"
    SENDER_TICK = "sender_tick"
    OPTIONAL_FEATURE = "optional_feature"
    VENDOR_PROBE = "vendor_probe"
    ROOT = "root"


def register_fatal(*types: type[BaseException]) -> None:
    for kind in types:
        if kind not in _fatal:
            _fatal.append(kind)


def register_translation_target(*types: type[Exception]) -> None:
    for kind in types:
        if kind not in _translation_targets:
            _translation_targets.append(kind)


def translation_targets() -> tuple[type[Exception], ...]:
    return tuple(_translation_targets)


def fatal_types() -> tuple[type[BaseException], ...]:
    return (*ALWAYS_PROPAGATE, *_fatal)


def _must_propagate(error: BaseException) -> bool:
    if not isinstance(error, Exception):
        return True
    if isinstance(error, fatal_types()):
        return True
    return isinstance(error, BaseExceptionGroup) and error.subgroup(fatal_types()) is not None


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _fields(log_fields: dict[str, str]) -> str:
    return "".join(f" {key}={value}" for key, value in sorted(log_fields.items()))


def _log_contained(boundary: Boundary, stage: str, error: BaseException, **log_fields: str) -> None:
    LOGGER.warning(
        "contained failure boundary=%s stage=%s exception_class=%s%s",
        boundary.value,
        stage,
        type(error).__name__,
        _fields(log_fields),
        exc_info=error,
    )


@dataclass(slots=True)
class Outcome:
    failed: bool = False
    error_class: str | None = None


@contextmanager
def isolate(boundary: Boundary, *, stage: str, **log_fields: str) -> Iterator[Outcome]:
    outcome = Outcome()
    try:
        yield outcome
    except BaseException as error:
        if _must_propagate(error):
            raise
        outcome.failed = True
        outcome.error_class = type(error).__name__
        _log_contained(boundary, stage, error, **log_fields)


def degrade(
    fn: Callable[[], T],
    *,
    stage: str,
    default: T,
    message: str | None = None,
    **log_fields: str,
) -> T:
    try:
        return fn()
    except BaseException as error:
        if _must_propagate(error):
            raise
        if message is None:
            _log_contained(Boundary.OPTIONAL_FEATURE, stage, error, **log_fields)
        else:
            LOGGER.warning("%s stage=%s%s", message, stage, _fields(log_fields), exc_info=error)
        return default


def _always_retry(error: BaseException) -> DeliveryDisposition:
    return DeliveryDisposition.RETRY


def attempt_delivery(
    fn: Callable[[], T],
    *,
    stage: str,
    code: str = "UNEXPECTED",
    classify: Callable[[BaseException], DeliveryDisposition] = _always_retry,
    **log_fields: str,
) -> T | DeliveryFailure:
    try:
        return fn()
    except BaseException as error:
        if _must_propagate(error):
            raise
        _log_contained(Boundary.SENDER_TICK, stage, error, **log_fields)
        return DeliveryFailure(classify(error), code, transport_error=_describe(error))


@dataclass(frozen=True, slots=True)
class ProbeFailure:
    reason: str


def probe(fn: Callable[[], T]) -> T | ProbeFailure:
    try:
        return fn()
    except BaseException as error:
        if _must_propagate(error):
            raise
        return ProbeFailure(reason=_describe(error))


def _primary_failure(primary: BaseException, cleanup: BaseException) -> BaseException:
    if _must_propagate(primary):
        return primary
    return cleanup if _must_propagate(cleanup) else primary


@contextmanager
def cleanup_on_failure(*cleanups: Callable[[], None]) -> Iterator[None]:
    try:
        yield
    except BaseException as primary:
        failure: BaseException = primary
        for cleanup in cleanups:
            try:
                cleanup()
            except BaseException as cleanup_error:
                chosen = _primary_failure(failure, cleanup_error)
                if chosen is failure:
                    failure.add_note(f"cleanup also failed: {_describe(cleanup_error)}")
                else:
                    chosen.__context__ = failure
                    failure = chosen
        if failure is primary:
            raise
        raise failure from primary


@contextmanager
def translate(to: type[E], message: str) -> Iterator[None]:
    if not issubclass(to, translation_targets()):
        raise TypeError(f"translate target {to.__name__} is not a registered translation target")
    try:
        yield
    except BaseException as error:
        if _must_propagate(error) or isinstance(error, to):
            raise
        raise to(message) from error


def root_sink(fn: Callable[[], int], *, on_error_exit_code: int, stage: str = "root") -> int:
    try:
        return fn()
    except SystemExit as exit_request:
        if exit_request.code is None:
            return 0
        return exit_request.code if isinstance(exit_request.code, int) else on_error_exit_code
    except BaseException as error:
        if _must_propagate(error):
            raise
        LOGGER.exception(
            "process root failed boundary=%s stage=%s exception_class=%s",
            Boundary.ROOT.value,
            stage,
            type(error).__name__,
        )
        return on_error_exit_code


__all__ = [
    "ALWAYS_PROPAGATE",
    "Boundary",
    "Outcome",
    "ProbeFailure",
    "attempt_delivery",
    "cleanup_on_failure",
    "degrade",
    "fatal_types",
    "isolate",
    "probe",
    "register_fatal",
    "register_translation_target",
    "root_sink",
    "translate",
    "translation_targets",
]
