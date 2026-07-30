from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

from contracts.decode_diagnostics import DecodeSelection
from contracts.frame import Frame
from edge.sources.rtsp_backend import (
    FALLBACK_EXTERNAL_ERRORS,
    NvdecRTSPBackend,
    ObservingRTSPBackend,
    OpenCVRTSPBackend,
    RTSPBackend,
)

_LivenessCallback = Callable[[str], None]
_OpenFailureCallback = Callable[[str], None]
_StopPredicate = Callable[[], bool]

_DEFAULT_PROCESSED_FPS = 5.0
RTSP_BACKEND_ENV = "ML_RTSP_BACKEND"
_DEFAULT_RTSP_BACKEND = "auto"


class RTSPSource:
    def __init__(
        self,
        url: str,
        max_failures: int = 30,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        reconnect_initial_backoff_sec: float = 0.25,
        reconnect_max_backoff_sec: float = 5.0,
        max_total_reconnects: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_wait: Callable[[float], bool] | None = None,
        stop_requested: _StopPredicate | None = None,
        backend: RTSPBackend | None = None,
        backend_name: str | None = None,
        target_fps: float = _DEFAULT_PROCESSED_FPS,
        clock: Callable[[], float] | None = None,
        pace_wait: Callable[[float], bool] | None = None,
        on_open_failure: _OpenFailureCallback | None = None,
    ) -> None:
        self._url = url
        self._max_failures = max(1, max_failures)
        self._open_timeout_ms = max(1, open_timeout_ms)
        self._read_timeout_ms = max(1, read_timeout_ms)
        self._backend = _create_backend(backend_name) if backend is None else backend
        self._reconnect_initial_backoff_sec = max(0.0, reconnect_initial_backoff_sec)
        self._reconnect_max_backoff_sec = max(
            self._reconnect_initial_backoff_sec,
            reconnect_max_backoff_sec,
        )
        self._max_total_reconnects = max_total_reconnects
        if self._max_total_reconnects is not None:
            self._max_total_reconnects = max(0, self._max_total_reconnects)
        self._sleep = sleep
        self._backoff_wait = backoff_wait
        self._stop_requested = stop_requested
        self._on_reconnecting: _LivenessCallback | None = None
        self._on_recovered: _LivenessCallback | None = None
        self._on_open_failure = on_open_failure
        # PM-3: 5 fps approximates the observed CPU-bound effective fall-inference
        # throughput before source pacing on the edge worker baseline; keeping that
        # cadence preserves model-window wall-clock scale without moving window/stride
        # out of the model artifact.
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._target_fps = target_fps
        self._frame_interval_sec = 1.0 / target_fps
        self._clock = time.monotonic if clock is None else clock
        self._pace_wait = pace_wait

    def set_liveness_callbacks(
        self,
        *,
        on_reconnecting: _LivenessCallback | None = None,
        on_recovered: _LivenessCallback | None = None,
        on_open_failure: _OpenFailureCallback | None = None,
    ) -> None:
        self._on_reconnecting = on_reconnecting
        self._on_recovered = on_recovered
        if on_open_failure is not None:
            self._on_open_failure = on_open_failure

    def __iter__(self) -> Iterator[Frame]:
        backend = self._backend
        capture = None
        t0: float | None = None
        frame_index = 0
        consecutive_failures = 0
        reconnects = 0
        reconnecting = False
        try:
            while not self._stop_is_requested():
                if capture is None:
                    try:
                        capture = backend.open(
                            self._url,
                            self._open_timeout_ms,
                            self._read_timeout_ms,
                        )
                    except FALLBACK_EXTERNAL_ERRORS:
                        self._notify_open_failure("spawn_failed")
                        if not reconnecting:
                            reconnecting = True
                            self._notify_reconnecting("open_failure")
                        if self._reconnect_budget_exhausted(reconnects):
                            break
                        reconnects += 1
                        if self._wait_for_backoff_or_stop(self._backoff_delay(reconnects)):
                            break
                        continue

                read_ok, image = backend.read(capture)
                if not read_ok or image is None:
                    consecutive_failures += 1
                    if consecutive_failures >= self._max_failures:
                        if not reconnecting:
                            reconnecting = True
                            self._notify_reconnecting("read_failure")
                        backend.release(capture)
                        capture = None
                        if self._reconnect_budget_exhausted(reconnects):
                            break
                        reconnects += 1
                        if self._wait_for_backoff_or_stop(self._backoff_delay(reconnects)):
                            break
                    continue
                consecutive_failures = 0
                if reconnecting:
                    reconnecting = False
                    self._notify_recovered("read_recovered")
                now = self._clock()
                if t0 is None:
                    t0 = now
                yield Frame(index=frame_index, time_sec=round(now - t0, 3), image=image)
                frame_index += 1
                remaining = max(0.0, self._frame_interval_sec - (self._clock() - now))
                if remaining > 0 and self._wait_for_pace_or_stop(remaining):
                    break
        finally:
            if capture is not None:
                backend.release(capture)

    def _reconnect_budget_exhausted(self, reconnects: int) -> bool:
        return (
            self._max_total_reconnects is not None
            and reconnects >= self._max_total_reconnects
        )

    def _backoff_delay(self, reconnects: int) -> float:
        if reconnects <= 0:
            return 0.0
        delay = self._reconnect_initial_backoff_sec * (2 ** min(reconnects - 1, 32))
        return min(delay, self._reconnect_max_backoff_sec)

    def _stop_is_requested(self) -> bool:
        return self._stop_requested is not None and self._stop_requested()

    def _wait_for_backoff_or_stop(self, delay_sec: float) -> bool:
        if self._stop_is_requested():
            return True
        if self._backoff_wait is not None:
            return self._backoff_wait(delay_sec) or self._stop_is_requested()
        self._sleep(delay_sec)
        return self._stop_is_requested()
    def _wait_for_pace_or_stop(self, delay_sec: float) -> bool:
        if self._stop_is_requested():
            return True
        if self._pace_wait is not None:
            return self._pace_wait(delay_sec) or self._stop_is_requested()
        self._sleep(delay_sec)
        return self._stop_is_requested()

    def _notify_reconnecting(self, reason: str) -> None:
        if self._on_reconnecting is not None:
            self._on_reconnecting(reason)

    def _notify_recovered(self, reason: str) -> None:
        if self._on_recovered is not None:
            self._on_recovered(reason)
    def _notify_open_failure(self, reason: str) -> None:
        if self._on_open_failure is not None:
            self._on_open_failure(reason)


def _normalize_backend_name(backend_name: str | None = None) -> str:
    raw_name = backend_name
    if raw_name is None:
        raw_name = os.environ.get(RTSP_BACKEND_ENV, _DEFAULT_RTSP_BACKEND)
    normalized = raw_name.strip().lower()
    return normalized or _DEFAULT_RTSP_BACKEND


def create_backend(
    backend_name: str | None = None,
    *,
    on_selection: Callable[[DecodeSelection], None] | None = None,
) -> RTSPBackend:
    """Resolve a decode backend by name (registry + fail-fast auto).

    - "auto" (default): NVDEC (GPU), FAIL LOUD on failure — no silent OpenCV
      fallback (ADR-0002). NVDEC failure propagates to the per-camera supervisor,
      which degrades that camera (DEGRADED); it never silently decodes on CPU.
    - "nvdec": force GPU decode (no fallback).
    - "opencv"/"cpu": force software CPU decode (explicit operator opt-in).
    """
    normalized = _normalize_backend_name(backend_name)
    if normalized == "auto":
        # ADR-0002: prefer GPU decode and fail loud; do NOT compose an OpenCV
        # safe-fallback. A downed NVDEC surfaces as a camera-level failure, not a
        # silent CPU degrade.
        backend = NvdecRTSPBackend()
        if on_selection is None:
            return backend
        return ObservingRTSPBackend(
            backend,
            requested=normalized,
            selected="nvdec",
            on_selection=on_selection,
        )
    factory = _BACKEND_FACTORIES.get(normalized)
    if factory is None:
        raise ValueError(
            "Unsupported RTSP backend "
            f"{normalized!r}; expected 'auto', 'nvdec', 'opencv', or 'cpu'."
        )
    backend = factory()
    if on_selection is None:
        return backend
    selected = "opencv" if normalized == "cpu" else normalized
    return ObservingRTSPBackend(
        backend,
        requested=normalized,
        selected=selected,
        on_selection=on_selection,
    )


# Named decode-backend registry (Strategy factories). "auto" is handled
# separately as a fallback composite over these.
_BACKEND_FACTORIES: dict[str, Callable[[], RTSPBackend]] = {
    "opencv": OpenCVRTSPBackend,
    "cpu": OpenCVRTSPBackend,
    "nvdec": NvdecRTSPBackend,
}


def _create_backend(backend_name: str | None = None) -> RTSPBackend:
    # Backward-compatible alias retained for existing callers/tests.
    return create_backend(backend_name)


__all__ = ["RTSPSource", "create_backend"]
