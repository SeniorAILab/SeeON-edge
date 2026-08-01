from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from worker.adapters.encode import FFmpegConcatFinalizer, Segment
from worker.types import BusinessEvent

_FFMPEG_BIN = shutil.which("ffmpeg")
_FFPROBE_BIN = shutil.which("ffprobe")


@pytest.mark.skipif(
    _FFMPEG_BIN is None or _FFPROBE_BIN is None,
    reason="ffmpeg and ffprobe are required for remux verification",
)
def test_finalizer_produces_faststart_h264_yuv420p_clip(tmp_path: Path) -> None:
    assert _FFMPEG_BIN is not None
    assert _FFPROBE_BIN is not None
    segment_dir = tmp_path / "segments"
    segment_dir.mkdir()
    _ = subprocess.run(
        (
            _FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=5:d=4",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            "-force_key_frames",
            "expr:gte(t,n_forced*2)",
            "-f",
            "segment",
            "-segment_time",
            "2",
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mp4",
            "-segment_format_options",
            "movflags=+faststart",
            str(segment_dir / "seg-%05d.mp4"),
        ),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    paths = tuple(sorted(segment_dir.glob("*.mp4")))
    assert len(paths) == 2
    segments = tuple(
        Segment(
            path=path,
            generation=1,
            start_time_sec=float(index * 2),
            end_time_sec=float((index + 1) * 2),
        )
        for index, path in enumerate(paths)
    )

    artifact = FFmpegConcatFinalizer(output_dir=tmp_path / "clip").finalize(
        segments,
        BusinessEvent("fall", "fall.detected", 7, "cam-1", "facility-1", 2.0, 0.91),
    )

    probe = subprocess.run(
        (
            _FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(artifact.path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "codec_name=h264" in probe.stdout
    assert "pix_fmt=yuv420p" in probe.stdout
    media = artifact.path.read_bytes()
    assert media.index(b"moov") < media.index(b"mdat")
