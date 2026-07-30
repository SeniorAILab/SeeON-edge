from __future__ import annotations

import json
import os
import shutil
import threading
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contracts.event import EventPayload
from contracts.frame import Frame
from edge.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from edge.evidence.evidence_outbox import ClipId
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.scheduler import Scheduler

DiskUsage = namedtuple("DiskUsage", "total used free")
EVENT_ONE = "00000000-0000-4000-8000-000000000001"
EVENT_TWO = "00000000-0000-4000-8000-000000000002"
EVENT_THREE = "00000000-0000-4000-8000-000000000003"


def test_retention_defaults_to_60_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLIP_STORE_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("CLIP_RETENTION_DAYS", raising=False)

    assert ClipRecorderConfig(store_dir=tmp_path).retention_days == 60


def test_clip_window_defaults_to_30_seconds_before_and_after(tmp_path: Path) -> None:
    config = ClipRecorderConfig(store_dir=tmp_path)

    assert config.pre_event_seconds == 30.0
    assert config.post_event_seconds == 30.0


@pytest.mark.parametrize(("configured", "expected"), (("59", 60), ("60", 60)))
def test_retention_environment_enforces_60_day_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: int,
) -> None:
    monkeypatch.setenv("CLIP_STORE_RETENTION_DAYS", configured)
    monkeypatch.delenv("CLIP_RETENTION_DAYS", raising=False)

    assert ClipRecorderConfig(store_dir=tmp_path).retention_days == expected


def test_direct_config_below_60_days_is_clamped(tmp_path: Path) -> None:
    config = ClipRecorderConfig(store_dir=tmp_path, retention_days=1)

    assert config.retention_days == 60


def _frame(index: int, time_sec: float | None = None) -> Frame:
    value = index % 255
    return Frame(
        index=index,
        time_sec=float(index) if time_sec is None else time_sec,
        image=np.full((16, 16, 3), value, dtype=np.uint8),
    )


def _low_disk_usage(_path: Path) -> DiskUsage:
    return DiskUsage(total=100, used=10, free=90)


def _recorder(
    tmp_path: Path,
    *,
    on_clip_finalized: Callable[[ClipId], None] | None = None,
    **overrides: object,
) -> ClipRecorder:
    # Default to a fake, always-low disk_usage_provider: _rotate() runs the
    # real shutil.disk_usage on every finalize, and treats the store as
    # over-watermark whenever the HOST machine's actual disk is more than
    # disk_high_watermark full — unrelated to anything this test wrote. On a
    # GitHub Actions runner (commonly >80% used out of the box from
    # preinstalled toolchains) that silently rmtree's the clip this test just
    # finalized moments earlier, while a spacious dev machine never crosses
    # the watermark and never notices. Tests that want to exercise real
    # disk-pressure eviction pass disk_usage_provider explicitly (see
    # test_clip_recorder_rotation_deletes_retention_and_oldest_over_disk_limit).
    disk_usage_provider = overrides.pop("disk_usage_provider", _low_disk_usage)
    config = ClipRecorderConfig(
        store_dir=tmp_path,
        segment_seconds=float(overrides.pop("segment_seconds", 2.0)),
        pre_event_seconds=float(overrides.pop("pre_event_seconds", 10.0)),
        post_event_seconds=float(overrides.pop("post_event_seconds", 20.0)),
        fps=float(overrides.pop("fps", 2.0)),
        retention_days=int(overrides.pop("retention_days", 60)),
        disk_high_watermark=float(overrides.pop("disk_high_watermark", 0.80)),
        max_queue_size=int(overrides.pop("max_queue_size", 128)),
        finalize_grace_seconds=float(overrides.pop("finalize_grace_seconds", 5.0)),
        stale_staging_seconds=float(overrides.pop("stale_staging_seconds", 3600.0)),
        rotate_min_interval_seconds=float(overrides.pop("rotate_min_interval_seconds", 30.0)),
    )
    assert overrides == {}
    return ClipRecorder(
        config,
        disk_usage_provider=disk_usage_provider,
        on_clip_finalized=on_clip_finalized,
    )


_REQUIRES_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="requires the ffmpeg binary for real-encode clip tests",
)


@_REQUIRES_FFMPEG
def test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window(tmp_path: Path) -> None:
    notifications: list[ClipId] = []

    def finalized(clip_id: ClipId) -> None:
        assert (tmp_path / "clips" / clip_id / "manifest.json").exists()
        assert not (tmp_path / "clips" / ".staging" / clip_id).exists()
        notifications.append(clip_id)

    recorder = _recorder(tmp_path, on_clip_finalized=finalized)
    recorder.start()
    try:
        clip_id: str | None = None
        for index in range(69):
            time_sec = index * 0.5
            assert recorder.on_frame("cam-1", _frame(index, time_sec))
            if time_sec == 12.0:
                clip_id = recorder.on_event("cam-1", EVENT_ONE)
                assert clip_id is not None
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    manifest_path = tmp_path / "clips" / clip_id / "manifest.json"
    video_path = tmp_path / "clips" / clip_id / "clip.mp4"
    assert manifest_path.exists()
    assert video_path.exists()
    assert not (tmp_path / "clips" / ".staging" / clip_id).exists()
    assert list((tmp_path / "clips" / clip_id).glob("*.tmp")) == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 2
    assert manifest["state"] == "READY"
    assert manifest["state_version"] == 2
    assert manifest["camera_id"] == "cam-1"
    assert manifest["clip_id"] == clip_id
    assert manifest["codec"] == "h264"
    assert manifest["mime_type"] == "video/mp4"
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_refs"] == [EVENT_ONE]
    assert manifest["finalized"] is True
    assert manifest["path"] == f"clips/{clip_id}/{video_path.name}"
    assert manifest["video_available"] is True
    assert len(manifest["sha256"]) == 64
    assert manifest["size_bytes"] == video_path.stat().st_size
    assert 1 <= manifest["duration_ms"] <= 120_000
    assert manifest["clip_start_at"].endswith("Z")
    assert manifest["clip_end_at"].endswith("Z")
    assert manifest["finalized_at"].endswith("Z")
    assert manifest["started_at"].endswith("Z")
    assert 29.0 <= manifest["duration_s"] <= 31.5
    assert recorder.stats.finalized_clips == 1
    assert notifications == [ClipId(clip_id)]


@_REQUIRES_FFMPEG
def test_clip_recorder_coalesces_event_refs_into_one_active_clip(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, pre_event_seconds=1.0, post_event_seconds=2.0)
    recorder.start()
    try:
        clip_ids: list[str] = []
        for index in range(16):
            assert recorder.on_frame("cam-1", _frame(index, index * 0.5))
            if index == 4:
                clip_id = recorder.on_event("cam-1", EVENT_ONE)
                assert clip_id is not None
                clip_ids.append(clip_id)
            if index == 6:
                clip_id = recorder.on_event("cam-1", EVENT_TWO)
                assert clip_id is not None
                clip_ids.append(clip_id)
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_ids[0] == clip_ids[1]
    manifest = json.loads(
        (tmp_path / "clips" / clip_ids[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_refs"] == [EVENT_ONE, EVENT_TWO]


def test_clip_recorder_keeps_coalesced_event_refs_ordered_and_unique(tmp_path: Path) -> None:
    recorder = _recorder(
        tmp_path, pre_event_seconds=0.2, post_event_seconds=0.4, fps=5.0
    )
    first_clip_id: str | None = None
    recorder.start()
    try:
        for index in range(8):
            assert recorder.on_frame("cam-1", _frame(index, index * 0.2))
            if index == 2:
                first_clip_id = recorder.on_event("cam-1", EVENT_ONE)
                assert first_clip_id is not None
                assert recorder.on_event("cam-1", EVENT_ONE) == first_clip_id
            if index == 3:
                assert first_clip_id is not None
                assert recorder.on_event("cam-1", EVENT_TWO) == first_clip_id
    finally:
        recorder.stop()

    assert first_clip_id is not None
    manifest = json.loads(
        (tmp_path / "clips" / first_clip_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_refs"] == [EVENT_ONE, EVENT_TWO]


def test_clip_recorder_drops_attach_only_event_after_queued_frame_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    @dataclass
    class _FileWriter:
        path: Path
        failed: bool = False
        error: str | None = None

        def write(self, _image: object) -> None:
            return

        def release(self) -> None:
            return

    def open_writer(path: Path, *_args: object) -> _FileWriter:
        path.write_bytes(b"clip-media")
        return _FileWriter(path)

    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", open_writer)
    monkeypatch.setattr(recorder_mod, "_append_video", lambda *_args: 1)
    recorder = _recorder(
        tmp_path,
        pre_event_seconds=0.0,
        post_event_seconds=1.0,
        segment_seconds=1.0,
    )
    event_handling_started = threading.Event()
    allow_first_event = threading.Event()
    first_finalized = threading.Event()
    allow_finalize_return = threading.Event()
    finalized_bytes: dict[str, bytes] = {}
    original_handle_event = recorder._handle_event
    original_finalize_clip = recorder._finalize_clip

    def pause_first_event(message: recorder_mod._EventMessage) -> None:
        if message.event_ref == EVENT_ONE:
            event_handling_started.set()
            assert allow_first_event.wait(timeout=1.0)
        original_handle_event(message)

    def pause_after_first_finalize(
        clip: recorder_mod._ActiveClip, **kwargs: object
    ) -> None:
        original_finalize_clip(clip, **kwargs)
        finalized_bytes["manifest"] = (clip.final_dir / "manifest.json").read_bytes()
        finalized_bytes["media"] = (clip.final_dir / "clip.mp4").read_bytes()
        first_finalized.set()
        assert allow_finalize_return.wait(timeout=1.0)

    monkeypatch.setattr(recorder, "_handle_event", pause_first_event)
    monkeypatch.setattr(recorder, "_finalize_clip", pause_after_first_finalize)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(0, 0.0))
        assert recorder.flush()
        clip_id = recorder.on_event("cam-1", EVENT_ONE)
        assert clip_id is not None
        assert event_handling_started.wait(timeout=1.0)
        assert recorder.on_frame("cam-1", _frame(1, 1.0))
        assert recorder.on_frame("cam-1", _frame(2, 2.0))
        assert recorder.on_event("cam-1", EVENT_TWO, allow_new_clip=False) == clip_id
        allow_first_event.set()
        assert first_finalized.wait(timeout=1.0)
        allow_finalize_return.set()
        assert recorder.flush()
    finally:
        allow_first_event.set()
        allow_finalize_return.set()
        recorder.stop()

    clip_dir = tmp_path / "clips" / clip_id
    assert recorder.stats.attach_missed_events == 1
    assert recorder.stats.finalized_clips == 1
    assert (clip_dir / "manifest.json").read_bytes() == finalized_bytes["manifest"]
    assert (clip_dir / "clip.mp4").read_bytes() == finalized_bytes["media"]


def test_clip_recorder_reserves_collision_free_id_and_preserves_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    @dataclass
    class _FileWriter:
        failed: bool = False
        error: str | None = None

        def write(self, _image: object) -> None:
            return

        def release(self) -> None:
            return

    original_dir = tmp_path / "clips" / "reused-id"
    original_dir.mkdir(parents=True)
    original_manifest = b'{"clip_id":"reused-id"}'
    original_media = b"existing-media"
    (original_dir / "manifest.json").write_bytes(original_manifest)
    (original_dir / "clip.mp4").write_bytes(original_media)
    clip_ids = iter(("reused-id", "fresh-id"))
    monkeypatch.setattr(recorder_mod, "_clip_id", lambda _camera_id: next(clip_ids))
    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", lambda *_args: _FileWriter())
    recorder = _recorder(tmp_path, pre_event_seconds=0.0, post_event_seconds=0.0)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(0, 0.0))
        assert recorder.flush()
        returned_clip_id = recorder.on_event("cam-1", EVENT_ONE)
        # Identity contract: the caller-visible id skips the colliding candidate
        # and equals the finalized artifact id.
        assert returned_clip_id == "fresh-id"
        assert recorder.flush()
    finally:
        recorder.stop()

    assert (original_dir / "manifest.json").read_bytes() == original_manifest
    assert (original_dir / "clip.mp4").read_bytes() == original_media
    fresh_manifest = tmp_path / "clips" / "fresh-id" / "manifest.json"
    assert fresh_manifest.exists()
    assert json.loads(fresh_manifest.read_text())["clip_id"] == "fresh-id"
    assert recorder.stats.clip_id_collisions == 1


def test_clip_recorder_selects_nvenc_when_probe_succeeds(monkeypatch) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "_resolved_encoder", None)
    monkeypatch.setattr(recorder_mod, "_probe_nvenc", lambda _bin: True)
    assert recorder_mod._resolve_encoder() == "h264_nvenc"
    # Resolution is cached per process: a later probe change does not re-resolve.
    monkeypatch.setattr(recorder_mod, "_probe_nvenc", lambda _bin: False)
    assert recorder_mod._resolve_encoder() == "h264_nvenc"


def test_clip_recorder_falls_back_to_libx264_when_nvenc_probe_fails(monkeypatch) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "_resolved_encoder", None)
    monkeypatch.setattr(recorder_mod, "_probe_nvenc", lambda _bin: False)
    assert recorder_mod._resolve_encoder() == "libx264"
def test_clip_recorder_nvenc_probe_uses_minimum_supported_resolution(monkeypatch) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    commands: list[list[str]] = []

    def _run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recorder_mod.subprocess, "run", _run)

    assert recorder_mod._probe_nvenc("ffmpeg")
    assert "color=c=black:s=256x256:r=25:d=0.08" in commands[0]


@dataclass
class _WriterSpy:
    failed: bool = False
    error: str | None = None
    released: bool = False

    def release(self) -> None:
        self.released = True
    def write(self, _image: object) -> None:
        return
def test_clip_recorder_stop_works_when_the_queue_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    recorder = _recorder(tmp_path, max_queue_size=1)
    handling_started = threading.Event()
    allow_handling_to_finish = threading.Event()

    def block_frame(_message: object) -> None:
        handling_started.set()
        allow_handling_to_finish.wait(timeout=1.0)

    monkeypatch.setattr(recorder, "_handle_frame", block_frame)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(0))
        assert handling_started.wait(timeout=1.0)
        assert recorder.on_frame("cam-1", _frame(1))
        assert recorder._queue.full()

        state = recorder._state("cam-1")
        writer = _WriterSpy()
        state.current = recorder_mod._OpenSegment(
            camera_id="cam-1",
            path=tmp_path / "segment.mp4",
            start_time_sec=0.0,
            started_at=datetime.now(UTC),
            writer=writer,
            codec="libx264",
        )

        recorder.stop(timeout=0.01)
        allow_handling_to_finish.set()
        assert recorder._thread is not None
        recorder._thread.join(timeout=1.0)
    finally:
        allow_handling_to_finish.set()
        recorder.stop()

    assert recorder._thread is not None and not recorder._thread.is_alive()
    assert writer.released

def test_clip_recorder_stop_drains_accepted_event_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", lambda *_args: _WriterSpy())
    recorder = _recorder(tmp_path)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(0))
        assert recorder.flush()
        clip_id = recorder.on_event("cam-1", EVENT_ONE)
        assert clip_id is not None
        recorder.stop()
    finally:
        recorder.stop()

    assert (tmp_path / "clips" / clip_id / "manifest.json").exists()


def test_clip_recorder_rolls_back_reserved_clip_id_after_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", lambda *_args: _WriterSpy())
    recorder = _recorder(tmp_path)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(0))
        assert recorder.flush()
        original_open_clip = recorder._open_clip

        def fail_open_clip(message, _state, _event_time_sec, _frame_size):
            staging_dir = tmp_path / "clips" / ".staging" / message.clip_id
            staging_dir.mkdir()
            raise OSError("staging unavailable")

        monkeypatch.setattr(recorder, "_open_clip", fail_open_clip)
        first_clip_id = recorder.on_event("cam-1", EVENT_ONE)
        assert first_clip_id is not None
        assert recorder.flush()
        assert recorder._clip_ids_by_camera.get("cam-1") is None
        assert not (tmp_path / "clips" / ".staging" / first_clip_id).exists()

        monkeypatch.setattr(recorder, "_open_clip", original_open_clip)
        second_clip_id = recorder.on_event("cam-1", EVENT_TWO)
        assert second_clip_id is not None
        assert second_clip_id != first_clip_id
        recorder.stop()
    finally:
        recorder.stop()

    assert (tmp_path / "clips" / second_clip_id / "manifest.json").exists()


def test_clip_recorder_coalesce_keeps_first_event_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    writer = _WriterSpy()
    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", lambda *_args: writer)
    recorder = _recorder(
        tmp_path,
        pre_event_seconds=0.0,
        post_event_seconds=5.0,
        segment_seconds=1.0,
        finalize_grace_seconds=0.0,
    )
    (tmp_path / "clips" / ".staging").mkdir(parents=True)
    state = recorder._state("cam-1")
    state.frame_size = (16, 16)
    state.last_time_sec = 2.0
    first = recorder_mod._EventMessage("cam-1", EVENT_ONE, "bed-exit", "clip-1")
    clip = recorder._open_clip(first, state, event_time_sec=2.0, frame_size=(16, 16))
    clip.opened_monotonic = 0.0
    recorder._active_clips.append(clip)
    recorder._update_active_clip_count()

    state.last_time_sec = 5.0
    monkeypatch.setattr(recorder_mod.time, "monotonic", lambda: 0.0)
    recorder._handle_event(recorder_mod._EventMessage("cam-1", EVENT_TWO, "bed-exit", "clip-1"))
    monkeypatch.setattr(recorder_mod.time, "monotonic", lambda: 8.0)
    recorder._finalize_ready_clips()

    assert clip.cutoff_time_sec == 7.0
    assert clip.force_finalize_extension_sec == 0.0
    assert writer.released is True


def test_clip_recorder_uses_registered_camera_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    opened_fps: list[float] = []
    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(
        recorder_mod,
        "_open_writer",
        lambda _path, _size, fps, _encoder: opened_fps.append(fps) or _WriterSpy(),
    )
    recorder = _recorder(tmp_path)
    recorder.set_camera_fps("cam-1", 15.0)

    recorder._open_writer_for_camera("cam-1", tmp_path, "clip", (16, 16))

    assert opened_fps == [15.0]



def test_clip_recorder_forces_finalize_after_stream_time_resets(
    tmp_path: Path, monkeypatch
) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    writer = _WriterSpy()
    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    monkeypatch.setattr(recorder_mod, "_open_writer", lambda *_args: writer)
    recorder = _recorder(
        tmp_path,
        pre_event_seconds=0.0,
        post_event_seconds=5.0,
        segment_seconds=1.0,
        finalize_grace_seconds=0.0,
    )
    (tmp_path / "clips" / ".staging").mkdir(parents=True)
    state = recorder._state("cam-1")
    state.frame_size = (16, 16)
    event = recorder_mod._EventMessage("cam-1", EVENT_ONE, "bed-exit", "clip-reset")
    clip = recorder._open_clip(event, state, event_time_sec=2.0, frame_size=(16, 16))
    recorder._active_clips.append(clip)
    recorder._update_active_clip_count()
    # RTSP stream time resets before post-event time is reached, but expiry must
    # still release the ffmpeg writer on the monotonic wall clock.
    clip.last_time_sec = 0.0
    clip.opened_monotonic = 0.0
    monkeypatch.setattr(recorder_mod.time, "monotonic", lambda: 8.0)

    recorder._finalize_ready_clips()

    assert writer.released
    assert recorder.stats.active_clips == 0
    assert recorder.stats.forced_finalized == 1
    assert (tmp_path / "clips" / "clip-reset" / "manifest.json").exists()
    assert not (tmp_path / "clips" / ".staging" / "clip-reset").exists()


def test_clip_recorder_start_sweeps_old_staging_directories(tmp_path: Path, monkeypatch) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    staging_root = tmp_path / "clips" / ".staging"
    old_staging = staging_root / "old"
    fresh_staging = staging_root / "fresh"
    old_staging.mkdir(parents=True)
    fresh_staging.mkdir(parents=True)
    old_time = datetime.now().timestamp() - 11
    os.utime(old_staging, (old_time, old_time))
    monkeypatch.setattr(recorder_mod, "_resolve_encoder", lambda: "libx264")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path, stale_staging_seconds=10.0),
        disk_usage_provider=_low_disk_usage,
        is_clip_held=lambda _clip_id: False,
    )
    recorder.start()
    try:
        assert not old_staging.exists()
        assert fresh_staging.exists()
        assert recorder.stats.stale_staging_cleaned == 1
        assert recorder.stats.encoder == "libx264"
    finally:
        recorder.stop()



def test_clip_recorder_ffmpeg_args_enforce_browser_playable_h264() -> None:
    import edge.evidence.clip_recorder as recorder_mod

    args = recorder_mod._ffmpeg_encode_args(
        "ffmpeg", Path("/tmp/clip.mp4"), (640, 360), 5.0, "h264_nvenc"
    )
    assert args[0] == "ffmpeg"
    # rawvideo BGR input contract for the raw-frame stdin pipe.
    assert "-f" in args and args[args.index("-f") + 1] == "rawvideo"
    assert "-pix_fmt" in args and "bgr24" in args
    assert "-i" in args and args[args.index("-i") + 1] == "pipe:0"
    # Browser-playable output contract, regardless of encoder.
    assert "-c:v" in args and args[args.index("-c:v") + 1] == "h264_nvenc"
    assert "-movflags" in args and "+faststart" in args
    assert "yuv420p" in args
    assert "640x360" in args
    assert args[-1] == "/tmp/clip.mp4"


@_REQUIRES_FFMPEG
def test_clip_recorder_writes_manifest_when_video_append_fails(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = _recorder(tmp_path, pre_event_seconds=0.0, post_event_seconds=1.0)
    sentinel = "TOP_SECRET /sensitive/path"

    def _fail_append(_path: Path, _writer) -> int:
        raise RuntimeError(sentinel)

    monkeypatch.setattr("edge.evidence.clip_recorder._append_video", _fail_append)

    recorder.start()
    try:
        clip_id: str | None = None
        for index in range(8):
            assert recorder.on_frame("cam-1", _frame(index, index * 0.5))
            if index == 2:
                clip_id = recorder.on_event("cam-1", EVENT_ONE, "bed-exit")
                assert clip_id is not None
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    manifest_path = tmp_path / "clips" / clip_id / "manifest.json"
    assert manifest_path.exists()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_type"] == "bed-exit"
    assert manifest["state"] == "UNAVAILABLE"
    assert manifest["reason_code"] == "ENCODER_FAILED"
    assert manifest["video_available"] is False
    assert manifest["path"] is None
    assert sentinel.encode() not in manifest_bytes
    assert "video_error" not in manifest
    assert recorder.stats.finalized_clips == 1
    assert recorder.stats.video_unavailable_clips == 1

def test_clip_recorder_writes_manifest_when_no_codec_opens(
    tmp_path: Path, monkeypatch
) -> None:
    # Reproduces the headless-CI/edge condition where no VideoWriter codec
    # initializes: the event must still produce a manifest so the event->clip
    # correlation survives (video_available=false).
    def _no_codec(_path, _frame_size, _fps, _codec):
        raise RuntimeError("no working codec")

    monkeypatch.setattr("edge.evidence.clip_recorder._open_writer", _no_codec)
    recorder = _recorder(tmp_path, pre_event_seconds=0.0, post_event_seconds=1.0)
    recorder.start()
    try:
        clip_id: str | None = None
        for index in range(8):
            recorder.on_frame("cam-1", _frame(index, index * 0.5))
            if index == 2:
                clip_id = recorder.on_event("cam-1", EVENT_ONE, "bed-exit")
                assert clip_id is not None
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    manifest_path = tmp_path / "clips" / clip_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_type"] == "bed-exit"
    assert manifest["state"] == "UNAVAILABLE"
    assert manifest["reason_code"] == "ENCODER_FAILED"
    assert manifest["video_available"] is False
    assert manifest["path"] is None


def test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = _recorder(tmp_path, pre_event_seconds=0.0, post_event_seconds=1.0)
    events: list[tuple[str, str, str]] = []
    real_replace = os.replace
    real_fsync = os.fsync
    real_rmtree = shutil.rmtree

    def _record_replace(source: Path, target: Path) -> None:
        events.append(("replace", Path(source).name, Path(target).name))
        real_replace(source, target)

    def _record_fsync(descriptor: int) -> None:
        target = Path(os.readlink(f"/proc/self/fd/{descriptor}")).name
        events.append(("fsync", target, ""))
        real_fsync(descriptor)

    def _record_rmtree(path: Path, *args: object, **kwargs: object) -> None:
        events.append(("rmtree", Path(path).name, ""))
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("edge.evidence.clip_recorder.os.replace", _record_replace)
    monkeypatch.setattr("edge.evidence.clip_recorder.os.fsync", _record_fsync)
    monkeypatch.setattr("edge.evidence.clip_recorder.shutil.rmtree", _record_rmtree)
    recorder.start()
    clip_id: str | None = None
    try:
        for index in range(8):
            assert recorder.on_frame("cam-1", _frame(index, index * 0.5))
            if index == 2:
                clip_id = recorder.on_event("cam-1", EVENT_THREE)
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    clips_dir_fsync = events.index(("fsync", "clips", ""))
    media_replace = events.index(("replace", "clip.tmp.mp4", "clip.mp4"))
    media_fsync = events.index(("fsync", "clip.mp4", ""))
    first_dir_fsync = events.index(("fsync", clip_id, ""), media_fsync)
    manifest_replace = events.index(("replace", "manifest.json.tmp", "manifest.json"))
    manifest_fsync = events.index(("fsync", "manifest.json", ""))
    second_dir_fsync = events.index(("fsync", clip_id, ""), first_dir_fsync + 1)
    staging_cleanup = events.index(("rmtree", clip_id, ""))
    staging_dir_fsync = events.index(("fsync", ".staging", ""), staging_cleanup)
    assert (
        clips_dir_fsync
        < media_replace
        < media_fsync
        < first_dir_fsync
        < manifest_replace
        < manifest_fsync
        < second_dir_fsync
        < staging_cleanup
        < staging_dir_fsync
    )
def test_clip_recorder_throttles_automatic_rotation(tmp_path: Path, monkeypatch) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    scans: list[Path] = []
    monkeypatch.setattr(
        recorder_mod,
        "_finalized_clips",
        lambda store_dir: scans.append(store_dir) or [],
    )
    monkeypatch.setattr(recorder_mod.time, "monotonic", lambda: 100.0)
    recorder = _recorder(tmp_path, rotate_min_interval_seconds=30.0)

    recorder._rotate()
    recorder._rotate()
    recorder._rotate(force=True)

    assert scans == [tmp_path, tmp_path]


def test_startup_checks_pressure_before_accepting_new_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: startup finds full storage with both young and actively held evidence.
    now = datetime.now(UTC)
    young_dir = _finalized_clip_dir(
        tmp_path,
        "young-startup",
        now - timedelta(days=61),
        finalized_at=now - timedelta(days=59),
    )
    held_dir = _finalized_clip_dir(
        tmp_path,
        "held-startup",
        now - timedelta(days=91),
        finalized_at=now - timedelta(days=90),
    )
    monkeypatch.setattr("edge.evidence.clip_recorder._resolve_encoder", lambda: "libx264")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path, retention_days=60, disk_high_watermark=0.80),
        disk_usage_provider=lambda _path: DiskUsage(total=100, used=95, free=5),
        is_clip_held=lambda clip_id: clip_id == held_dir.name,
    )

    # When: the recorder starts and immediately receives a new event.
    recorder.start()
    try:
        assert recorder.stats.recording_suspended is True
        new_clip_id = recorder.on_event("cam-1", "event-under-startup-pressure")
    finally:
        recorder.stop()

    # Then: startup pressure is synchronous, and protected evidence remains intact.
    assert new_clip_id is None
    assert young_dir.exists()
    assert held_dir.exists()



def test_clip_recorder_rotation_uses_finalized_at_at_60_day_boundary(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    boundary_dir = _finalized_clip_dir(
        tmp_path,
        "boundary",
        now - timedelta(days=61),
        finalized_at=now - timedelta(days=60),
    )
    young_dir = _finalized_clip_dir(
        tmp_path,
        "young",
        now - timedelta(days=61),
        finalized_at=now - timedelta(days=59),
    )
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path, retention_days=60, disk_high_watermark=0.80),
        disk_usage_provider=lambda _path: DiskUsage(total=100, used=90, free=10),
        is_clip_held=lambda _clip_id: False,
    )
    recorder.start()
    try:
        assert recorder.rotate_once()
    finally:
        recorder.stop()

    assert not boundary_dir.exists()
    assert young_dir.exists()
    assert recorder.stats.recording_suspended is True


def test_legacy_retention_clock_uses_manifest_mtime_not_started_at(tmp_path: Path) -> None:
    import edge.evidence.clip_recorder as recorder_mod

    now = datetime.now(UTC)
    legacy_dir = _finalized_clip_dir(tmp_path, "legacy", now - timedelta(days=90))
    manifest_path = legacy_dir / "manifest.json"
    conservative_time = now - timedelta(days=10)
    os.utime(manifest_path, (conservative_time.timestamp(), conservative_time.timestamp()))

    clips = recorder_mod._finalized_clips(tmp_path)

    assert clips == [(datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC), legacy_dir)]


def test_clip_recorder_queue_full_drops_without_blocking(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, max_queue_size=1)

    assert recorder.on_frame("cam-1", _frame(0))
    assert not recorder.on_frame("cam-1", _frame(1))
    assert recorder.dropped_frame_count == 1


@dataclass(slots=True)
class _Sink:
    events: list[EventPayload] = field(default_factory=list)

    def emit(self, event: EventPayload) -> None:
        self.events.append(event)


@dataclass(slots=True)
class _RecorderSpy:
    frames: list[tuple[str, int]] = field(default_factory=list)
    events: list[tuple[str, str]] = field(default_factory=list)
    event_types: list[str | None] = field(default_factory=list)

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        self.frames.append((camera_id, frame.index))
        return True

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str:
        del allow_new_clip
        self.events.append((camera_id, event_ref))
        self.event_types.append(event_type)
        return "clip-id"


class _Detector:
    def update(self, _observation, time_sec: float | None = None) -> EventPayload:
        return {"event_type": "other", "event_id": f"evt-{time_sec}"}


def test_camera_worker_records_frames_and_admitted_events_without_blocking() -> None:
    sink = _Sink()
    recorder = _RecorderSpy()
    worker = CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=(),
        runners={},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        event_sink=sink,
        clip_recorder=recorder,
    )

    worker.process_frame(_frame(7, 7.0))

    assert recorder.frames == [("cam-1", 7)]
    assert recorder.events == [
        ("cam-1", str(sink.events[0]["edge_event_id"]))
    ]
    assert recorder.event_types == ["other"]
    assert sink.events[0]["camera_id"] == "cam-1"
@dataclass
class _IncidentManagerSpy:
    admitted_at: list[float] = field(default_factory=list)

    def admit(self, _event: object, *, now_sec: float) -> bool:
        self.admitted_at.append(now_sec)
        return True


def test_camera_worker_uses_monotonic_time_for_incident_admission(monkeypatch) -> None:
    import edge.runtime.camera_worker as camera_worker_mod

    manager = _IncidentManagerSpy()
    monkeypatch.setattr(camera_worker_mod.time, "monotonic", lambda: 987.0)
    worker = CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=(),
        runners={},
        scheduler=Scheduler({}),
        domain_detectors=(_Detector(),),
        incident_manager=manager,  # type: ignore[arg-type]
    )

    worker.process_frame(_frame(7, 0.0))

    assert manager.admitted_at == [987.0]



def _finalized_clip_dir(
    root: Path,
    clip_id: str,
    started_at: datetime,
    *,
    finalized_at: datetime | None = None,
) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "cam-1",
                "event_ref": f"evt-{clip_id}",
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "duration_s": 30.0,
                "codec": "mp4v",
                "path": f"clips/{clip_id}/clip.mp4",
                "finalized": True,
                "video_available": True,
            }
        ),
        encoding="utf-8",
    )
    if finalized_at is not None:
        manifest_path = clip_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["manifest_schema_version"] = 2
        payload["finalized_at"] = finalized_at.isoformat().replace("+00:00", "Z")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return clip_dir
