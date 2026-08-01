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
from worker.adapters.decode.nvdec_cuvid.probe import (
    NvdecCapability,
    probe_nvdec_cuvid_capability,
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
        return f'{{"streams":[{{"width":1920,"height":1080,"codec_name":"{codec_name}"}}]}}'

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


def test_nvdec_capability_true_via_hwaccels_short_circuits_before_decoders() -> None:
    """Matches edge/runtime/profile/registry.py:118-119's short-circuit: a
    hwaccels match returns True without ever querying -decoders."""
    # Given
    calls: list[tuple[str, ...]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        calls.append(args)
        return "Hardware acceleration methods:\ncuda\nvdpau\n"

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability == NvdecCapability(True, "ffmpeg CUDA/CUVID hwaccel is available")
    assert calls == [("ffmpeg", "-hide_banner", "-hwaccels")]


def test_nvdec_capability_true_via_decoders_when_hwaccels_has_no_marker() -> None:
    # Given
    calls: list[tuple[str, ...]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        calls.append(args)
        if args[-1] == "-hwaccels":
            return "Hardware acceleration methods:\nvideotoolbox\n"
        return "V..... h264_cuvid            Nvidia CUVID H264 decoder\n"

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability == NvdecCapability(True, "ffmpeg CUVID decoder is available")
    assert calls == [
        ("ffmpeg", "-hide_banner", "-hwaccels"),
        ("ffmpeg", "-hide_banner", "-decoders"),
    ]


def test_nvdec_capability_false_when_neither_hwaccel_nor_decoder_present() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        if args[-1] == "-hwaccels":
            return "Hardware acceleration methods:\nvideotoolbox\nopencl\nvulkan\n"
        return "V..... h264                  H264 / AVC / MPEG-4 AVC\n"

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then -- fail closed: this repo's macOS dev machines hit exactly this path
    # (real `ffmpeg -hwaccels`/`-decoders` output on this host has neither).
    assert capability == NvdecCapability(
        False, "ffmpeg has no CUDA/CUVID/NVDEC hwaccel or *_cuvid decoder"
    )


def test_nvdec_capability_matches_cuvid_decoder_by_word_boundary_not_substring() -> None:
    # Given -- "cuvid" appears in running text but no `*_cuvid` decoder token
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        if args[-1] == "-hwaccels":
            return "Hardware acceleration methods:\nvideotoolbox\n"
        return "This build was compiled without cuvid support.\n"

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability.available is False


def test_nvdec_capability_false_when_ffmpeg_binary_missing() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise FileNotFoundError("no such file: ffmpeg")

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then -- fail closed, never raises past the probe boundary
    assert capability == NvdecCapability(False, "ffmpeg missing")


def test_nvdec_capability_false_when_ffmpeg_query_times_out() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        raise subprocess.TimeoutExpired(args, timeout_sec)

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability == NvdecCapability(False, "ffmpeg capability probe timed out")


def test_nvdec_capability_false_when_ffmpeg_query_exits_non_zero() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        raise subprocess.CalledProcessError(1, args)

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability == NvdecCapability(False, "ffmpeg capability probe failed with exit code 1")


def test_nvdec_capability_false_when_ffmpeg_query_raises_unexpected_error() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise ValueError("unexpected")

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then -- decode probe must never break startup, even on an unmodeled error
    assert capability == NvdecCapability(False, "ffmpeg capability probe failed: ValueError")


def test_nvdec_capability_false_when_decoder_query_times_out_after_hwaccels_miss() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        if args[-1] == "-hwaccels":
            return "Hardware acceleration methods:\nvideotoolbox\n"
        raise subprocess.TimeoutExpired(args, timeout_sec)

    # When
    capability = probe_nvdec_cuvid_capability(runner=runner)

    # Then
    assert capability == NvdecCapability(False, "ffmpeg decoder probe timed out")
