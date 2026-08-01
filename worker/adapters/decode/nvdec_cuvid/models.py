from __future__ import annotations

from dataclasses import dataclass

from worker.adapters.decode.nvdec_cuvid.errors import NvdecConfigError


@dataclass(frozen=True, slots=True)
class NvdecCuvidConfig:
    camera_id: str
    url: str
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    ffmpeg_bin: str = "ffmpeg"

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise NvdecConfigError("camera id must not be blank")
        if not self.url.strip():
            raise NvdecConfigError("RTSP URL must not be blank")
        if self.open_timeout_ms <= 0:
            raise NvdecConfigError("open timeout must be positive")
        if self.read_timeout_ms <= 0:
            raise NvdecConfigError("read timeout must be positive")
        if not self.ffmpeg_bin.strip():
            raise NvdecConfigError("ffmpeg binary must not be blank")


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    width: int
    height: int
    codec_name: str


__all__ = ["NvdecCuvidConfig", "StreamMetadata"]
