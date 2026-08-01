from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Final, final

import numpy as np

from contracts.frame import Frame
from worker.adapters.decode.nvdec_cuvid.errors import (
    NvdecReadError,
    sanitized_nvdec_error,
)
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata
from worker.adapters.decode.nvdec_cuvid.probe import (
    ProbeRunner,
    cuvid_decoder_for,
    probe_stream_metadata,
    run_ffprobe,
)
from worker.adapters.decode.nvdec_cuvid.process import (
    DecoderProcess,
    ProcessSpawner,
    spawn_decoder_process,
)
from worker.types import FramePacket

_RGB_CHANNELS: Final = 3
_PROCESS_REAP_TIMEOUT_SEC: Final = 5.0


def ffmpeg_decode_args(
    config: NvdecCuvidConfig,
    decoder: str,
) -> tuple[str, ...]:
    return (
        config.ffmpeg_bin,
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-hwaccel",
        "cuda",
        "-c:v",
        decoder,
        "-i",
        config.url,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    )


@final
class NvdecCuvidSession:
    """Own mutable frame timing and one decoder process for a camera."""

    def __init__(
        self,
        config: NvdecCuvidConfig,
        metadata: StreamMetadata,
        process: DecoderProcess,
        clock: Callable[[], float],
    ) -> None:
        self._config = config
        self._metadata = metadata
        self._process = process
        self._clock = clock
        self._frame_size = metadata.width * metadata.height * _RGB_CHANNELS
        self._seq = 0
        self._first_frame_at: float | None = None
        self._closed = False
        self._close_lock = threading.Lock()

    def read(self) -> FramePacket | None:
        if self._closed:
            return None
        started_at = self._clock()
        payload = self._process.read_frame(self._config.read_timeout_ms / 1000.0)
        finished_at = self._clock()
        if payload is None:
            return None
        if len(payload) != self._frame_size:
            self.close()
            raise NvdecReadError(self._frame_size, len(payload))
        image = np.frombuffer(payload, dtype=np.uint8).reshape(
            self._metadata.height,
            self._metadata.width,
            _RGB_CHANNELS,
        ).copy()
        if self._first_frame_at is None:
            self._first_frame_at = finished_at
        pts = max(0.0, finished_at - self._first_frame_at)
        decode_time_ms = max(0.0, (finished_at - started_at) * 1000.0)
        seq = self._seq
        self._seq += 1
        frame = Frame(index=seq, time_sec=pts, image=image)
        return FramePacket(
            camera_id=self._config.camera_id,
            frame=frame,
            pts=pts,
            seq=seq,
            width=self._metadata.width,
            height=self._metadata.height,
            decode_time_ms=decode_time_ms,
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        _ = self._process.reap(timeout_sec=_PROCESS_REAP_TIMEOUT_SEC)


@final
class NvdecCuvidAdapter:
    def __init__(
        self,
        *,
        probe_runner: ProbeRunner = run_ffprobe,
        process_spawner: ProcessSpawner = spawn_decoder_process,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe_runner = probe_runner
        self._process_spawner = process_spawner
        self._clock = clock

    def open(self, config: NvdecCuvidConfig) -> NvdecCuvidSession:
        metadata = probe_stream_metadata(config, runner=self._probe_runner)
        decoder = cuvid_decoder_for(metadata.codec_name)
        frame_size = metadata.width * metadata.height * _RGB_CHANNELS
        try:
            process = self._process_spawner(
                ffmpeg_decode_args(config, decoder),
                frame_size,
            )
        except OSError as error:
            raise sanitized_nvdec_error("ffmpeg spawn failed", error) from None
        return NvdecCuvidSession(config, metadata, process, self._clock)


__all__ = [
    "NvdecCuvidAdapter",
    "NvdecCuvidSession",
    "ffmpeg_decode_args",
]
