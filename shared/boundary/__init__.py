import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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


DEFAULT_LOG_INTERVAL_SECONDS: Final = 300.0


@dataclass(slots=True)
class LogThrottle:
    interval_seconds: float = DEFAULT_LOG_INTERVAL_SECONDS
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    last_logged_at: float | None = None

    def failed(self) -> bool:
        self.failures += 1
        now = self.clock()
        if self.last_logged_at is not None and now - self.last_logged_at < self.interval_seconds:
            return False
        self.last_logged_at = now
        return True

    def succeeded(self) -> int:
        failures = self.failures
        self.failures = 0
        self.last_logged_at = None
        return failures


def _log_traceback(boundary: Boundary, stage: str, error: BaseException, fields: str) -> None:
    LOGGER.debug(
        "contained failure traceback boundary=%s stage=%s%s",
        boundary.value,
        stage,
        fields,
        exc_info=error,
    )


def _log_contained(
    boundary: Boundary,
    stage: str,
    error: BaseException,
    *,
    throttle: LogThrottle | None = None,
    level: int = logging.WARNING,
    message: str | None = None,
    **log_fields: str,
) -> None:
    fields = _fields(log_fields)
    if throttle is None:
        emit, first = True, True
    else:
        emit = throttle.failed()
        first = throttle.failures == 1
        fields = f"{fields} failures={throttle.failures}"
    if emit:
        if message is None:
            LOGGER.log(
                level,
                "contained failure boundary=%s stage=%s exception_class=%s%s",
                boundary.value,
                stage,
                type(error).__name__,
                fields,
            )
        else:
            LOGGER.log(level, "%s stage=%s%s", message, stage, fields)
    if first:
        _log_traceback(boundary, stage, error, fields)


def _log_recovered(boundary: Boundary, stage: str, throttle: LogThrottle | None) -> None:
    if throttle is None:
        return
    failures = throttle.succeeded()
    if failures:
        LOGGER.info(
            "contained failure recovered boundary=%s stage=%s failures=%d",
            boundary.value,
            stage,
            failures,
        )


@dataclass(slots=True)
class Outcome:
    failed: bool = False
    error_class: str | None = None


@contextmanager
def isolate(
    boundary: Boundary,
    *,
    stage: str,
    throttle: LogThrottle | None = None,
    level: int = logging.WARNING,
    **log_fields: str,
) -> Iterator[Outcome]:
    outcome = Outcome()
    try:
        yield outcome
    except BaseException as error:
        if _must_propagate(error):
            raise
        outcome.failed = True
        outcome.error_class = type(error).__name__
        _log_contained(boundary, stage, error, throttle=throttle, level=level, **log_fields)
    else:
        _log_recovered(boundary, stage, throttle)


def degrade(
    fn: Callable[[], T],
    *,
    stage: str,
    default: T,
    message: str | None = None,
    throttle: LogThrottle | None = None,
    level: int = logging.WARNING,
    **log_fields: str,
) -> T:
    try:
        value = fn()
    except BaseException as error:
        if _must_propagate(error):
            raise
        _log_contained(
            Boundary.OPTIONAL_FEATURE,
            stage,
            error,
            throttle=throttle,
            level=level,
            message=message,
            **log_fields,
        )
        return default
    _log_recovered(Boundary.OPTIONAL_FEATURE, stage, throttle)
    return value


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
    error: BaseException | None = field(default=None, compare=False, repr=False)


def probe(fn: Callable[[], T]) -> T | ProbeFailure:
    try:
        return fn()
    except BaseException as error:
        if _must_propagate(error):
            raise
        return ProbeFailure(reason=_describe(error), error=error)


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


def root_sink(
    fn: Callable[[], int],
    *,
    on_error_exit_code: int,
    stage: str = "root",
    none_exit_code: int = 0,
) -> int:
    try:
        return fn()
    except SystemExit as exit_request:
        if exit_request.code is None:
            return none_exit_code
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
    "DEFAULT_LOG_INTERVAL_SECONDS",
    "Boundary",
    "LogThrottle",
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
