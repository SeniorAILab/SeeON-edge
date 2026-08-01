from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

from worker.adapters.decode.nvdec_cuvid import (
    NvdecCuvidConfig,
    NvdecUnavailableError,
    UnsupportedCodecError,
    cuvid_decoder_for,
    probe_stream_metadata,
)

ProbeRunner = Callable[[tuple[str, ...], float], str]


@pytest.mark.parametrize(
    ("codec_name", "expected_decoder"),
    [("h264", "h264_cuvid"), ("hevc", "hevc_cuvid")],
)
def test_probe_selects_cuvid_decoder_for_supported_stream(
    codec_name: str,
    expected_decoder: str,
) -> None:
    # Given
    calls: list[tuple[tuple[str, ...], float]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        calls.append((args, timeout_sec))
        return (
            '{"streams":[{"width":1920,"height":1080,"codec_name":"'
            f'{codec_name}"}}]}}'
        )

    config = NvdecCuvidConfig(
        camera_id="camera-a",
        url="rtsp://camera.local/live",
        open_timeout_ms=2500,
        read_timeout_ms=1000,
        ffmpeg_bin="/opt/ffmpeg/bin/ffmpeg",
    )

    # When
    metadata = probe_stream_metadata(config, runner=runner)

    # Then
    assert (metadata.width, metadata.height, metadata.codec_name) == (
        1920,
        1080,
        codec_name,
    )
    assert cuvid_decoder_for(metadata.codec_name) == expected_decoder
    assert calls == [
        (
            (
                "/opt/ffmpeg/bin/ffprobe",
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
                "rtsp://camera.local/live",
            ),
            2.5,
        )
    ]


def test_unsupported_codec_fails_loud_without_decoder_selection() -> None:
    # Given
    unsupported_codec = "mpeg4"

    # When / Then
    with pytest.raises(UnsupportedCodecError) as error:
        _ = cuvid_decoder_for(unsupported_codec)

    assert error.value.codec_name == unsupported_codec
    assert "mpeg4" in str(error.value)


def test_ffprobe_failure_masks_command_url_and_credentials() -> None:
    # Given
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        raise subprocess.CalledProcessError(7, args, stderr=raw_url)

    config = NvdecCuvidConfig(camera_id="camera-a", url=raw_url)

    # When
    with pytest.raises(NvdecUnavailableError) as error:
        _ = probe_stream_metadata(config, runner=runner)

    # Then
    assert str(error.value) == "ffprobe failed: CalledProcessError (returncode=7)"
    assert raw_url not in str(error.value)
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)
    assert error.value.__cause__ is None


def test_ffprobe_rejects_unusable_metadata_without_echoing_payload() -> None:
    # Given
    secret_payload = '{"streams":[{"codec_name":"secret-token"}]}'

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return secret_payload

    config = NvdecCuvidConfig(camera_id="camera-a", url="rtsp://camera.local/live")

    # When
    with pytest.raises(NvdecUnavailableError) as error:
        _ = probe_stream_metadata(config, runner=runner)

    # Then
    assert str(error.value).startswith("ffprobe metadata unusable:")
    assert "secret-token" not in str(error.value)
