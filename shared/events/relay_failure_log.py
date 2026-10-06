from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, final

from shared.events.evidence_export_contract import DeliveryFailure

DEFAULT_SUMMARY_INTERVAL_SEC: Final = 60.0


class RelayFailureClass(StrEnum):
    TRANSPORT = "transport"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"


_CLIENT_ERROR_HINTS: Final[dict[int, str]] = {
    401: "check relay token",
    403: "check runtime enrollment / auth",
}
_DEFAULT_CLIENT_HINT: Final = "check worker relay config"
_DEFAULT_TRANSPORT_HINT: Final = "cannot reach relay host; will keep retrying"


def _server_error_hint(status: int | None) -> str:
    if status is None:
        return "edge API rejected the request; will keep retrying"
    return f"edge API returned {status}; will keep retrying"


@dataclass(frozen=True, slots=True)
class RelayFailureOutcome:
    failure_class: RelayFailureClass
    reason: str
    hint: str


def classify_relay_failure(failure: DeliveryFailure) -> RelayFailureOutcome:
    if failure.transport_error is not None:
        return RelayFailureOutcome(
            RelayFailureClass.TRANSPORT, failure.transport_error, _DEFAULT_TRANSPORT_HINT
        )
    status = failure.status_code
    if status is None:
        return RelayFailureOutcome(
            RelayFailureClass.SERVER_ERROR, failure.code, _server_error_hint(status)
        )
    if 400 <= status <= 499:
        return RelayFailureOutcome(
            RelayFailureClass.CLIENT_ERROR,
            str(status),
            _CLIENT_ERROR_HINTS.get(status, _DEFAULT_CLIENT_HINT),
        )
    return RelayFailureOutcome(
        RelayFailureClass.SERVER_ERROR, str(status), _server_error_hint(status)
    )


@dataclass(slots=True)
class _ActiveFailure:
    outcome: RelayFailureOutcome
    attempt: int
    first_seen: float
    last_logged: float
    since_last_summary: int


@final
class RelayFailureLog:
    def __init__(
        self,
        logger: logging.Logger,
        *,
        channel: str,
        method: str,
        summary_interval_sec: float = DEFAULT_SUMMARY_INTERVAL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._channel = channel
        self._method = method
        self._summary_interval_sec = summary_interval_sec
        self._clock = clock
        self._active: _ActiveFailure | None = None

    def record_failure(self, failure: DeliveryFailure, *, path: str) -> None:
        outcome = classify_relay_failure(failure)
        now = self._clock()
        active = self._active
        if active is None or active.outcome.failure_class != outcome.failure_class:
            self._active = _ActiveFailure(outcome, 1, now, now, 0)
            self._logger.log(
                _level(outcome.failure_class),
                "relay %s %s -> %s (%s: %s) [%s, attempt 1]",
                self._method,
                path,
                outcome.reason,
                outcome.failure_class.value,
                outcome.hint,
                self._channel,
                extra={
                    "relay_channel": self._channel,
                    "relay_status_code": failure.status_code,
                    "relay_failure_class": outcome.failure_class.value,
                },
            )
            return
        active.attempt += 1
        active.since_last_summary += 1
        elapsed = now - active.last_logged
        if elapsed < self._summary_interval_sec:
            return
        self._logger.log(
            _level(outcome.failure_class),
            "relay %s %s still failing: %s (%s: %s) "
            "[%s, attempt %d, %d occurrence(s) in the last %.0fs]",
            self._method,
            path,
            outcome.reason,
            outcome.failure_class.value,
            outcome.hint,
            self._channel,
            active.attempt,
            active.since_last_summary,
            elapsed,
            extra={
                "relay_channel": self._channel,
                "relay_status_code": failure.status_code,
                "relay_failure_class": outcome.failure_class.value,
            },
        )
        active.last_logged = now
        active.since_last_summary = 0

    def record_success(self, *, path: str) -> None:
        active = self._active
        if active is None:
            return
        now = self._clock()
        self._logger.info(
            "relay %s %s recovered after %d failed attempt(s) over %.1fs [%s]",
            self._method,
            path,
            active.attempt,
            now - active.first_seen,
            self._channel,
            extra={"relay_channel": self._channel},
        )
        self._active = None


def _level(failure_class: RelayFailureClass) -> int:
    return logging.ERROR if failure_class is RelayFailureClass.CLIENT_ERROR else logging.WARNING


__all__ = [
    "DEFAULT_SUMMARY_INTERVAL_SEC",
    "RelayFailureClass",
    "RelayFailureLog",
    "RelayFailureOutcome",
    "classify_relay_failure",
]
