from __future__ import annotations

import subprocess
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worker.adapters.decode.nvdec_cuvid.errors import (
    UnsupportedCodecError,
    sanitized_nvdec_error,
)
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata

_CUVID_DECODER_BY_CODEC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hevc": "hevc_cuvid",
        "h265": "hevc_cuvid",
        "h264": "h264_cuvid",
        "avc": "h264_cuvid",
        "av1": "av1_cuvid",
        "vp9": "vp9_cuvid",
        "vp8": "vp8_cuvid",
        "mjpeg": "mjpeg_cuvid",
    }
)


class ProbeRunner(Protocol):
    def __call__(self, args: tuple[str, ...], timeout_sec: float, /) -> str: ...


class _FFprobeStream(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec_name: str = Field(min_length=1)


class _FFprobeDocument(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    streams: tuple[_FFprobeStream, ...] = Field(min_length=1)


def ffprobe_binary(ffmpeg_bin: str) -> str:
    if ffmpeg_bin.endswith("ffmpeg"):
        return f"{ffmpeg_bin[:-len('ffmpeg')]}ffprobe"
    return "ffprobe"


def ffprobe_args(config: NvdecCuvidConfig) -> tuple[str, ...]:
    return (
        ffprobe_binary(config.ffmpeg_bin),
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
        config.url,
    )


def run_ffprobe(args: tuple[str, ...], timeout_sec: float) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return completed.stdout


def probe_stream_metadata(
    config: NvdecCuvidConfig,
    *,
    runner: ProbeRunner = run_ffprobe,
) -> StreamMetadata:
    try:
        stdout = runner(ffprobe_args(config), config.open_timeout_ms / 1000.0)
    except (OSError, subprocess.SubprocessError) as error:
        raise sanitized_nvdec_error("ffprobe failed", error) from None
    try:
        stream = _FFprobeDocument.model_validate_json(stdout).streams[0]
    except ValidationError as error:
        raise sanitized_nvdec_error("ffprobe metadata unusable", error) from None
    return StreamMetadata(stream.width, stream.height, stream.codec_name)


def cuvid_decoder_for(codec_name: str) -> str:
    normalized = codec_name.strip().lower()
    try:
        return _CUVID_DECODER_BY_CODEC[normalized]
    except KeyError:
        raise UnsupportedCodecError(normalized) from None


__all__ = [
    "ProbeRunner",
    "cuvid_decoder_for",
    "ffprobe_args",
    "ffprobe_binary",
    "probe_stream_metadata",
    "run_ffprobe",
]
