from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NoReturn

from numpy.typing import NDArray

from contracts.decode_diagnostics import DecodeSelection
from edge.sources.rtsp import _normalize_backend_name, create_backend
from edge.sources.rtsp_backend import RTSPBackend, RTSPCapture
from edge.sources.rtsp_url import mask_rtsp_url

ProbeErrorClass = Literal["timeout", "decode", "auth"]


@dataclass(frozen=True)
class RTSPProbeResult:
    masked_url: str
    requested_backend: str
    backend: str
    width: int
    height: int
    channels: int

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "url": self.masked_url,
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


class RTSPProbeError(RuntimeError):
    def __init__(
        self,
        error_class: ProbeErrorClass,
        message: str,
        masked_url: str,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.masked_url = masked_url

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": False,
            "url": self.masked_url,
            "error_class": self.error_class,
            "message": str(self),
        }


def probe_first_frame(
    url: str,
    *,
    backend: RTSPBackend | None = None,
    backend_name: str | None = None,
    timeout_ms: int = 5000,
    open_timeout_ms: int | None = None,
    read_timeout_ms: int | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> RTSPProbeResult:
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    masked_url = mask_rtsp_url(url)
    requested_backend = (
        _normalize_backend_name(backend_name) if backend is None else "injected"
    )
    selected_backend_name = "injected"

    if backend is None:
        def on_selection(selection: DecodeSelection) -> None:
            nonlocal selected_backend_name
            selected_backend_name = selection.selected

        selected_backend = create_backend(
            requested_backend,
            on_selection=on_selection,
        )
    else:
        selected_backend = backend
    open_timeout = _positive_timeout(open_timeout_ms, timeout_ms)
    read_timeout = _positive_timeout(read_timeout_ms, timeout_ms)
    deadline = monotonic() + (timeout_ms / 1000.0)
    capture: RTSPCapture | None = None

    try:
        capture = selected_backend.open(url, open_timeout, read_timeout)
        while True:
            if monotonic() >= deadline:
                _raise_timeout(masked_url, timeout_ms)
            try:
                read_ok, frame = selected_backend.read(capture)
            except Exception as exc:  # noqa: BLE001 - normalize backend exceptions.
                raise _probe_error(_classify_exception(exc), masked_url, timeout_ms) from exc
            if not read_ok:
                if monotonic() >= deadline:
                    _raise_timeout(masked_url, timeout_ms)
                continue
            height, width, channels = _frame_dimensions(frame, masked_url)
            return RTSPProbeResult(
                masked_url=masked_url,
                requested_backend=requested_backend,
                backend=selected_backend_name,
                width=width,
                height=height,
                channels=channels,
            )
    except RTSPProbeError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize backend exceptions.
        raise _probe_error(_classify_exception(exc), masked_url, timeout_ms) from exc
    finally:
        if capture is not None:
            selected_backend.release(capture)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe an RTSP URL by decoding its first frame."
    )
    parser.add_argument("url")
    parser.add_argument(
        "--backend",
        choices=("auto", "nvdec", "opencv", "cpu"),
        default=None,
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--open-timeout-ms", type=int, default=None)
    parser.add_argument("--read-timeout-ms", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        result = probe_first_frame(
            args.url,
            backend_name=args.backend,
            timeout_ms=args.timeout_ms,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
        )
    except RTSPProbeError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


def _positive_timeout(value: int | None, default: int) -> int:
    if value is None:
        return default
    return max(1, value)


def _frame_dimensions(
    frame: NDArray[object] | None,
    masked_url: str,
) -> tuple[int, int, int]:
    if frame is None:
        raise _probe_error("decode", masked_url, 0)
    if frame.ndim == 2:
        height, width = frame.shape
        channels = 1
    elif frame.ndim == 3:
        height, width, channels = frame.shape
    else:
        raise _probe_error("decode", masked_url, 0)
    if height <= 0 or width <= 0 or channels <= 0:
        raise _probe_error("decode", masked_url, 0)
    return int(height), int(width), int(channels)




def _classify_exception(exc: Exception) -> ProbeErrorClass:
    message = str(exc).lower()
    if any(
        term in message
        for term in ("401", "403", "unauthorized", "forbidden", "auth")
    ):
        return "auth"
    if any(term in message for term in ("timeout", "timed out")):
        return "timeout"
    return "decode"


def _raise_timeout(masked_url: str, timeout_ms: int) -> NoReturn:
    raise _probe_error("timeout", masked_url, timeout_ms)


def _probe_error(
    error_class: ProbeErrorClass,
    masked_url: str,
    timeout_ms: int,
) -> RTSPProbeError:
    if error_class == "timeout":
        return RTSPProbeError(
            "timeout",
            "Timed out waiting for first decodable RTSP frame "
            f"from {masked_url} within {timeout_ms} ms.",
            masked_url,
        )
    if error_class == "auth":
        return RTSPProbeError(
            "auth",
            f"RTSP authentication failed for {masked_url}; "
            "verify username, password, and NVR permissions.",
            masked_url,
        )
    return RTSPProbeError(
        "decode",
        f"Could not decode a first RTSP frame from {masked_url}; "
        "verify codec, stream profile, and URL path.",
        masked_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProbeErrorClass",
    "RTSPProbeError",
    "RTSPProbeResult",
    "main",
    "mask_rtsp_url",
    "probe_first_frame",
]
