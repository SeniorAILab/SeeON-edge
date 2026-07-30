from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from typing import IO, Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.decode_diagnostics import DECODE_FALLBACK_REASONS, DecodeSelection
from edge.sources.rtsp_url import mask_rtsp_url

LOGGER = logging.getLogger(__name__)
_FallbackSelectionSink = Callable[[DecodeSelection], None]


@runtime_checkable
class RTSPCapture(Protocol):
    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...


@runtime_checkable
class RTSPBackend(Protocol):
    """Decode RTSP streams into RGB uint8 frames."""
    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> RTSPCapture: ...

    # Returns (ok, frame) where frame is RGB NDArray[np.uint8] when ok is True.
    def read(self, capture: RTSPCapture) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def release(self, capture: RTSPCapture) -> None: ...


class OpenCVRTSPBackend:
    """Software (CPU) RTSP decode via OpenCV's FFmpeg backend."""

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> RTSPCapture:
        _ensure_rtsp_over_tcp()
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            read_timeout_ms,
        ]
        try:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            try:
                capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            except TypeError:
                capture = cv2.VideoCapture(url)
        set_capture_property(capture, "CAP_PROP_BUFFERSIZE", 1)
        set_capture_property(capture, "CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_ms)
        return capture

    def read(self, capture: RTSPCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        read_ok, frame_bgr = capture.read()
        if not read_ok or frame_bgr is None:
            return read_ok, None
        return True, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self, capture: RTSPCapture) -> None:
        capture.release()


# ─── NVDEC (GPU hardware) backend ─────────────────────────────────────────────
# H.265/H.264 NVR streams are decoded on the NVIDIA GPU's dedicated NVDEC engine
# via an FFmpeg subprocess (`-hwaccel cuda -c:v <codec>_cuvid`). This offloads
# software decode from the CPU and — unlike the CPU path — reliably reconstructs
# reference frames at full frame rate (no "Could not find ref with POC" flood),
# so the live wall can run smoothly. Frames are emitted as rgb24 for the
# CPU-side detection pipeline (the win is the decode, not zero-copy).

_FFMPEG_BIN_ENV = "ML_FFMPEG_BIN"
_DEFAULT_FFMPEG_BIN = "ffmpeg"
_CUVID_DECODER_BY_CODEC = {
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "h264": "h264_cuvid",
    "avc": "h264_cuvid",
    "av1": "av1_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
    "mjpeg": "mjpeg_cuvid",
}


class NvdecUnavailableError(RuntimeError):
    """Raised when an NVDEC capture cannot be established (probe/spawn failure)."""


def _ffmpeg_bin() -> str:
    return os.environ.get(_FFMPEG_BIN_ENV, _DEFAULT_FFMPEG_BIN).strip() or _DEFAULT_FFMPEG_BIN


def _ffprobe_bin() -> str:
    binary = _ffmpeg_bin()
    if binary.endswith("ffmpeg"):
        return f"{binary[: -len('ffmpeg')]}ffprobe"
    return "ffprobe"


def _probe_stream_metadata(url: str, timeout_sec: float) -> tuple[int, int, str]:
    """Return (width, height, codec_name) for the first video stream via ffprobe."""
    cmd = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name",
        "-of",
        "json",
        url,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_sec),
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _sanitized_nvdec_error("ffprobe failed", exc) from None
    try:
        streams = json.loads(completed.stdout).get("streams", [])
        stream = streams[0]
        return int(stream["width"]), int(stream["height"]), str(stream["codec_name"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise _sanitized_nvdec_error("ffprobe metadata unusable", exc) from None


def _sanitized_nvdec_error(stage: str, exc: BaseException) -> NvdecUnavailableError:
    returncode = getattr(exc, "returncode", None)
    bounded_returncode = returncode if isinstance(returncode, int) else None
    return NvdecUnavailableError(
        f"{stage}: {type(exc).__name__} (returncode={bounded_returncode})"
    )


class _NvdecCapture:
    """RTSPCapture backed by an FFmpeg NVDEC decode subprocess (rgb24 stdout)."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        width: int,
        height: int,
        read_timeout_ms: int,
    ) -> None:
        self._proc: subprocess.Popen[bytes] | None = proc
        self.width = width
        self.height = height
        self._frame_size = width * height * 3
        self._read_timeout_sec = max(1, read_timeout_ms) / 1000.0
        self._chunks: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=2)
        self._pending = bytearray()
        self._stop_reader = threading.Event()
        self._release_lock = threading.Lock()
        self._reader_thread = threading.Thread(
            target=self._pump_stdout,
            args=(proc.stdout,),
            name="nvdec-stdout-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._proc is None:
            return False, None
        deadline = time.monotonic() + self._read_timeout_sec
        while len(self._pending) < self._frame_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._release(timeout_sec=0.05)
                return False, None
            try:
                chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                self._release(timeout_sec=0.05)
                return False, None
            if isinstance(chunk, BaseException) or chunk is None:
                self._release(timeout_sec=0.05)
                return False, None
            self._pending.extend(chunk)
        buffer = bytes(self._pending[: self._frame_size])
        del self._pending[: self._frame_size]
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame.copy()  # copy: frombuffer is read-only; overlays draw in place

    def _pump_stdout(self, stream: IO[bytes] | None) -> None:
        if stream is None:
            self._put_chunk(None)
            return
        try:
            while not self._stop_reader.is_set():
                chunk = stream.read(self._frame_size)
                if not chunk:
                    self._put_chunk(None)
                    return
                self._put_chunk(chunk)
        except BaseException as exc:  # noqa: BLE001 - pipe failures are read failures
            self._put_chunk(exc)

    def _put_chunk(self, chunk: bytes | BaseException | None) -> None:
        while not self._stop_reader.is_set():
            try:
                self._chunks.put(chunk, timeout=0.05)
            except queue.Full:
                continue
            else:
                return

    def set(self, prop_id: int, value: float) -> bool:  # noqa: ARG002 - protocol parity
        return False

    def release(self) -> None:
        self._release(timeout_sec=5)

    def _release(self, *, timeout_sec: float) -> None:
        with self._release_lock:
            proc = self._proc
            self._proc = None
            self._stop_reader.set()
        if proc is None:
            return
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                pass
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        self._reader_thread.join(timeout=timeout_sec)


class NvdecRTSPBackend:
    """GPU (NVIDIA NVDEC) RTSP decode via an FFmpeg cuvid subprocess."""

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> RTSPCapture:
        width, height, codec = _probe_stream_metadata(url, max(1, open_timeout_ms) / 1000.0)
        args = [
            _ffmpeg_bin(),
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-hwaccel",
            "cuda",
        ]
        decoder = _CUVID_DECODER_BY_CODEC.get(codec.strip().lower())
        if decoder is not None:
            args += ["-c:v", decoder]
        args += ["-i", url, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise _sanitized_nvdec_error("ffmpeg spawn failed", exc) from None
        return _NvdecCapture(proc, width, height, read_timeout_ms)

    def read(self, capture: RTSPCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: RTSPCapture) -> None:
        capture.release()


# ─── Fallback (auto) backend ──────────────────────────────────────────────────


class _FallbackCapture:
    """Wraps the chosen backend's capture; replays the validating first frame."""

    def __init__(
        self,
        backend: RTSPBackend,
        capture: RTSPCapture,
        *,
        backend_name: str,
        pending: NDArray[np.uint8] | None,
    ) -> None:
        self._backend = backend
        self._capture = capture
        self.backend_name = backend_name
        self._pending = pending

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._pending is not None:
            frame = self._pending
            self._pending = None
            return True, frame
        return self._backend.read(self._capture)

    def set(self, prop_id: int, value: float) -> bool:  # noqa: ARG002 - protocol parity
        return False

    def release(self) -> None:
        self._backend.release(self._capture)


class _ObservingCapture:
    def __init__(self, backend: RTSPBackend, capture: RTSPCapture, backend_name: str) -> None:
        self._backend = backend
        self._capture = capture
        self.backend_name = backend_name

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        return self._backend.read(self._capture)

    def set(self, prop_id: int, value: float) -> bool:
        return self._capture.set(prop_id, value)

    def release(self) -> None:
        self._backend.release(self._capture)


class ObservingRTSPBackend:
    """Record a forced backend selection without changing its decode behavior."""

    def __init__(
        self,
        backend: RTSPBackend,
        *,
        requested: str,
        selected: str,
        on_selection: _FallbackSelectionSink,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend
        self._requested = requested
        self._selected = selected
        self._on_selection = on_selection
        self._clock = clock

    def open(
        self, url: str, open_timeout_ms: int, read_timeout_ms: int
    ) -> RTSPCapture:
        capture = self._backend.open(url, open_timeout_ms, read_timeout_ms)
        self._on_selection(
            DecodeSelection(
                requested=self._requested,
                selected=self._selected,
                fallback_count=0,
                last_reason=None,
                updated_at_sec=self._clock(),
            )
        )
        return _ObservingCapture(self._backend, capture, self._selected)

    def read(self, capture: RTSPCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: RTSPCapture) -> None:
        capture.release()


class FallbackRTSPBackend:
    """Try preferred backends in order, fall back to the last safe backend."""

    def __init__(
        self,
        backends: list[tuple[str, RTSPBackend]],
        *,
        requested: str = "auto",
        on_selection: _FallbackSelectionSink | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not backends:
            raise ValueError("FallbackRTSPBackend requires at least one backend")
        self._backends = backends
        self._requested = requested
        self._on_selection = on_selection
        self._clock = clock
        self._fallback_count = 0
        self._last_reason: str | None = None

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> RTSPCapture:
        masked_url = mask_rtsp_url(url)
        *preferred, safe = self._backends
        for name, backend in preferred:
            try:
                capture = backend.open(url, open_timeout_ms, read_timeout_ms)
            except FALLBACK_EXTERNAL_ERRORS as exc:
                self._record_fallback(_fallback_reason(exc, stage="open"))
                LOGGER.warning(
                    "decode backend %s unavailable (%s) for %s",
                    name,
                    self._last_reason,
                    masked_url,
                )
                continue
            try:
                ok, frame = backend.read(capture)
            except FALLBACK_EXTERNAL_ERRORS as exc:
                self._record_fallback(_fallback_reason(exc, stage="read"))
                LOGGER.warning(
                    "decode backend %s failed first frame (%s) for %s",
                    name,
                    self._last_reason,
                    masked_url,
                )
                backend.release(capture)
                continue
            if ok and frame is not None:
                self._record_selection(name)
                LOGGER.info("decode backend active: %s", name)
                return _FallbackCapture(
                    backend, capture, backend_name=name, pending=frame
                )
            self._record_fallback("first_frame_failed")
            LOGGER.warning(
                "decode backend %s produced no first frame (%s) for %s",
                name,
                self._last_reason,
                masked_url,
            )
            backend.release(capture)
        safe_name, safe_backend = safe
        capture = safe_backend.open(url, open_timeout_ms, read_timeout_ms)
        self._record_selection(safe_name)
        LOGGER.info("decode backend active: %s (fallback)", safe_name)
        return _FallbackCapture(
            safe_backend, capture, backend_name=safe_name, pending=None
        )

    def read(self, capture: RTSPCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: RTSPCapture) -> None:
        capture.release()

    def _record_fallback(self, reason: str) -> None:
        if reason not in DECODE_FALLBACK_REASONS:
            raise ValueError(f"unsupported decode fallback reason: {reason}")
        self._fallback_count += 1
        self._last_reason = reason

    def _record_selection(self, selected: str) -> None:
        if self._on_selection is None:
            return
        self._on_selection(
            DecodeSelection(
                requested=self._requested,
                selected=selected,
                fallback_count=self._fallback_count,
                last_reason=self._last_reason,
                updated_at_sec=self._clock(),
            )
        )


FALLBACK_EXTERNAL_ERRORS = (
    NvdecUnavailableError,
    OSError,
    subprocess.SubprocessError,
    cv2.error,
)


def _fallback_reason(exc: BaseException, *, stage: str) -> str:
    if stage == "read":
        return "first_frame_failed"
    if isinstance(exc, NvdecUnavailableError):
        message = str(exc).lower()
        if "metadata unusable" in message:
            return "unsupported_codec"
        if "ffprobe" in message:
            return "ffprobe_unavailable"
    return "spawn_failed"


def set_capture_property(capture: RTSPCapture, name: str, value: int) -> None:
    prop = getattr(cv2, name, None)
    if prop is not None:
        capture.set(prop, value)


_RTSP_OVER_TCP_OPTION = "rtsp_transport;tcp"


def _ensure_rtsp_over_tcp() -> None:
    """Default OpenCV's FFmpeg RTSP transport to TCP.

    Live H.265 (HEVC) NVR substreams over the default UDP transport drop
    packets, corrupting reference frames and flooding the decoder with
    "First slice in a frame missing" / "Could not find ref with POC" /
    "Error constructing the frame RPS" errors that degrade detection. TCP
    trades a little latency for a lossless stream. Operators can override
    the whole option string via OPENCV_FFMPEG_CAPTURE_OPTIONS.
    """
    if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_OVER_TCP_OPTION


__all__ = [
    "FALLBACK_EXTERNAL_ERRORS",
    "FallbackRTSPBackend",
    "NvdecRTSPBackend",
    "NvdecUnavailableError",
    "OpenCVRTSPBackend",
    "ObservingRTSPBackend",
    "RTSPBackend",
    "RTSPCapture",
    "set_capture_property",
]
