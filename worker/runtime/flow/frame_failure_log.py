from __future__ import annotations

import logging
import time

_FAILURE_LOG_INTERVAL_SEC = 30.0


class PolicyFrameError(RuntimeError):
    pass


class ThrottledFailureLog:
    def __init__(self, logger: logging.Logger, camera_id: str) -> None:
        self._logger = logger
        self._camera_id = camera_id
        self._logged_at: float | None = None
        self._suppressed = 0

    def record(self, error: PolicyFrameError) -> None:
        now = time.monotonic()
        first = self._logged_at is None
        if not first and now - self._logged_at < _FAILURE_LOG_INTERVAL_SEC:
            self._suppressed += 1
            return
        suppressed, self._suppressed = self._suppressed, 0
        self._logged_at = now
        self._logger.warning(
            "native policy frame failed: camera_id=%s cause=%r suppressed_since_last_log=%d",
            self._camera_id,
            error.__cause__,
            suppressed,
            exc_info=error if first else None,
        )


class LogOnce:
    def __init__(self, logger: logging.Logger, message: str, camera_id: str) -> None:
        self._logger = logger
        self._message = message
        self._camera_id = camera_id
        self._logged = False

    def record(self) -> None:
        if self._logged:
            return
        self._logged = True
        self._logger.warning(self._message, self._camera_id, exc_info=True)
