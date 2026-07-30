from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import cv2

from contracts.frame import Frame
from edge.evidence.clip_store_lock import ClipStoreLock
from edge.evidence.evidence_manifest import (
    ClipEvidenceError,
    ClipManifest,
    finalize_ready_manifest,
    unavailable_manifest,
)
from edge.evidence.evidence_outbox_types import ClipId, EdgeEventId, EvidenceReasonCode
from edge.evidence.evidence_retention import DiskUsage, EvidenceRetention, PurgeCandidate

LOGGER = logging.getLogger(__name__)

CLIP_STORE_DIR_ENV = "CLIP_STORE_DIR"
CLIP_STORE_RETENTION_DAYS_ENV = "CLIP_STORE_RETENTION_DAYS"
CLIP_RETENTION_DAYS_ENV = "CLIP_RETENTION_DAYS"
CLIP_STORE_MAX_USAGE_ENV = "CLIP_STORE_MAX_USAGE"
CLIP_DISK_HIGH_WATERMARK_ENV = "CLIP_DISK_HIGH_WATERMARK"

DEFAULT_CLIP_STORE_DIR = "/var/lib/clip-store"
MIN_RETENTION_DAYS = 60
DEFAULT_RETENTION_DAYS = MIN_RETENTION_DAYS
DEFAULT_DISK_HIGH_WATERMARK = 0.80

CLIP_FFMPEG_BIN_ENV = "CLIP_FFMPEG_BIN"
DEFAULT_FFMPEG_BIN = "ffmpeg"
NVENC_ENCODER = "h264_nvenc"
SOFTWARE_ENCODER = "libx264"
DEFAULT_FINALIZE_GRACE_SECONDS = 5.0
DEFAULT_STALE_STAGING_SECONDS = 60.0 * 60.0
DEFAULT_ROTATE_MIN_INTERVAL_SECONDS = 30.0
NVENC_MIN_PROBE_DIMENSION = 256

# Resolved once per process: the FFmpeg H.264 encoder used for every clip.
# h264_nvenc (NVIDIA hardware) is preferred on the edge GPU; libx264 (software)
# is the CI / no-GPU fallback. Both always produce H.264 + yuv420p + faststart
# MP4 so the front <video> tag can play clips in any browser.
_encoder_lock = threading.Lock()
_resolved_encoder: str | None = None



@dataclass(frozen=True, slots=True)
class ClipRecorderConfig:
    store_dir: Path = field(
        default_factory=lambda: Path(_env_str(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))
    )
    segment_seconds: float = 2.0
    pre_event_seconds: float = 30.0
    post_event_seconds: float = 30.0
    fps: float = 5.0
    retention_days: int = field(default_factory=lambda: _env_retention_days())
    disk_high_watermark: float = field(default_factory=lambda: _env_disk_high_watermark())
    max_queue_size: int = 128
    codec: str = "mp4v"
    finalize_grace_seconds: float = DEFAULT_FINALIZE_GRACE_SECONDS
    stale_staging_seconds: float = DEFAULT_STALE_STAGING_SECONDS
    rotate_min_interval_seconds: float = DEFAULT_ROTATE_MIN_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "retention_days", max(MIN_RETENTION_DAYS, self.retention_days)
        )


@dataclass(slots=True)
class ClipRecorderStats:
    dropped_frames: int = 0
    dropped_events: int = 0
    attach_missed_events: int = 0
    clip_id_collisions: int = 0
    finalized_clips: int = 0
    failed_writes: int = 0
    video_unavailable_clips: int = 0
    active_clips: int = 0
    forced_finalized: int = 0
    stale_staging_cleaned: int = 0
    encoder: str | None = None
    held_clips: int = 0
    purge_failures: int = 0
    recording_suspended: bool = False


@dataclass(frozen=True, slots=True)
class _CodecSpec:
    codec: str
    suffix: str


# The clip encoder is FFmpeg (see _resolve_encoder). cv2.VideoWriter is not used:
# on headless CI/edge builds OpenCV cannot emit browser-playable H.264 (avc1
# resolves to h264_v4l2m2m, which fails without a V4L2 device). FFmpeg with
# h264_nvenc (edge GPU) or libx264 (fallback) produces a real H.264 MP4 instead.
# cv2 is still used for frame color conversion and segment decode on the
# append path, never for encoding.


@dataclass(frozen=True, slots=True)
class _FrameMessage:
    camera_id: str
    frame: Frame


@dataclass(frozen=True, slots=True)
class _EventMessage:
    camera_id: str
    event_ref: str
    event_type: str | None
    clip_id: str
    allow_new_clip: bool = True


@dataclass(frozen=True, slots=True)
class _RotateMessage:
    done: threading.Event | None = None


class _ClipIdCollisionError(RuntimeError):
    """A reserved clip id unexpectedly has existing on-disk output."""

_Message = _FrameMessage | _EventMessage | _RotateMessage


@dataclass(slots=True)
class _Segment:
    camera_id: str
    path: Path
    start_time_sec: float
    end_time_sec: float
    started_at: datetime
    frame_count: int


@dataclass(slots=True)
class _OpenSegment:
    camera_id: str
    path: Path
    start_time_sec: float
    started_at: datetime
    writer: _FfmpegWriter
    codec: str
    frame_count: int = 0
    end_time_sec: float = 0.0


@dataclass(slots=True)
class _CameraState:
    camera_id: str
    segment_dir: Path
    current: _OpenSegment | None = None
    closed_segments: list[_Segment] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None
    last_time_sec: float | None = None


@dataclass(slots=True)
class _ActiveClip:
    camera_id: str
    event_ref: str
    event_type: str | None
    clip_id: str
    event_time_sec: float
    cutoff_time_sec: float
    staging_dir: Path
    final_dir: Path
    tmp_video_path: Path
    final_video_path: Path | None
    manifest_path: Path
    writer: _FfmpegWriter | None
    codec: str | None
    started_at: datetime
    start_time_sec: float
    last_time_sec: float
    opened_monotonic: float = field(default_factory=time.monotonic)
    video_error: str | None = None
    appended_paths: set[Path] = field(default_factory=set)
    event_refs: list[str] = field(default_factory=list)
    frame_count: int = 0
    force_finalize_extension_sec: float = 0.0


class ClipRecorder:
    """Disk-backed edge evidence recorder.

    Frames and event notifications enter through a bounded queue so camera inference never
    waits for disk, encoding, rotation, or finalize work. The recorder keeps per-camera
    rolling segment files on disk and assembles event clips from those segment files.
    """

    def __init__(
        self,
        config: ClipRecorderConfig | None = None,
        *,
        disk_usage_provider: Callable[[Path], DiskUsage] | None = None,
        is_clip_held: Callable[[str], bool] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> None:
        self.config = ClipRecorderConfig() if config is None else config
        self.stats = ClipRecorderStats()
        self._queue: queue.Queue[_Message] = queue.Queue(maxsize=self.config.max_queue_size)
        self._thread: threading.Thread | None = None
        self._states: dict[str, _CameraState] = {}
        self._active_clips: list[_ActiveClip] = []
        self._retention = EvidenceRetention(
            self.config.store_dir,
            is_held=(lambda _clip_id: True) if is_clip_held is None else is_clip_held,
            disk_usage_provider=(
                shutil.disk_usage if disk_usage_provider is None else disk_usage_provider
            ),
        )
        self._store_lock: ClipStoreLock | None = None
        self._lock = threading.Lock()
        self._codec_by_camera: dict[str, _CodecSpec] = {}
        self._fps_by_camera: dict[str, float] = {}
        self._last_rotate_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._clip_ids_by_camera: dict[str, str] = {}
        self._accepting_messages = True
        self._startup_hook = startup_hook
        self._on_clip_finalized = on_clip_finalized

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        if fps <= 0:
            raise ValueError("camera fps must be > 0")
        self._fps_by_camera[camera_id] = fps

    @classmethod
    def from_env(
        cls,
        *,
        is_clip_held: Callable[[str], bool] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> ClipRecorder:
        return cls(
            ClipRecorderConfig(),
            is_clip_held=is_clip_held,
            startup_hook=startup_hook,
            on_clip_finalized=on_clip_finalized,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._accepting_messages = False
        store_lock = ClipStoreLock.acquire(self.config.store_dir)
        lock_retained = False
        try:
            if self._startup_hook is not None:
                self._startup_hook()
            (self.config.store_dir / "segments").mkdir(parents=True, exist_ok=True)
            (self.config.store_dir / "clips" / ".staging").mkdir(parents=True, exist_ok=True)
            self._sweep_stale_staging()
            self._rotate(force=True)
            with self._lock:
                self.stats.encoder = _resolve_encoder()
                self._accepting_messages = True
            self._store_lock = store_lock
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="clip-recorder", daemon=True)
            self._thread.start()
            lock_retained = True
        finally:
            if not lock_retained:
                self._store_lock = None
                store_lock.close()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            self._accepting_messages = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._thread is None or not self._thread.is_alive():
            if self._store_lock is not None:
                self._store_lock.close()
                self._store_lock = None

    def flush(self, *, timeout: float = 5.0) -> bool:
        done = threading.Event()
        if not self._put_control(_RotateMessage(done)):
            return False
        return done.wait(timeout)

    def rotate_once(self, *, timeout: float = 5.0) -> bool:
        return self.flush(timeout=timeout)

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        with self._lock:
            if not self._accepting_messages:
                return False
            try:
                self._queue.put_nowait(_FrameMessage(camera_id=camera_id, frame=frame))
            except queue.Full:
                self.stats.dropped_frames += 1
                return False
        return True

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        with self._lock:
            if not self._accepting_messages:
                return None
            clip_id = self._clip_ids_by_camera.get(camera_id)
            created_clip_id = clip_id is None
            if clip_id is None:
                if self.stats.recording_suspended:
                    return None
                if not allow_new_clip:
                    return None
                clip_id = self._reserve_clip_id(camera_id)
                self._clip_ids_by_camera[camera_id] = clip_id
            try:
                self._queue.put_nowait(
                    _EventMessage(
                        camera_id=camera_id,
                        event_ref=event_ref,
                        event_type=event_type,
                        clip_id=clip_id,
                        allow_new_clip=allow_new_clip,
                    )
                )
            except queue.Full:
                self.stats.dropped_events += 1
                if created_clip_id and self._clip_ids_by_camera.get(camera_id) == clip_id:
                    self._clip_ids_by_camera.pop(camera_id, None)
                return None
        return clip_id

    @property
    def dropped_frame_count(self) -> int:
        return self.stats.dropped_frames

    @property
    def dropped_event_count(self) -> int:
        return self.stats.dropped_events
    @property
    def active_clips(self) -> int:
        with self._lock:
            return self.stats.active_clips

    def _put_control(self, message: _RotateMessage) -> bool:
        with self._lock:
            if not self._accepting_messages:
                return False
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                return False
        return True

    def _run(self) -> None:
        try:
            while True:
                try:
                    message = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._finalize_ready_clips()
                    except Exception as exc:  # noqa: BLE001 - recorder failures must not kill inference
                        LOGGER.warning("clip recorder timeout finalize failed: %s", exc)
                        with self._lock:
                            self.stats.failed_writes += 1
                    continue
                try:
                    if isinstance(message, _FrameMessage):
                        self._handle_frame(message)
                    elif isinstance(message, _EventMessage):
                        self._handle_event(message)
                    elif isinstance(message, _RotateMessage):
                        self._rotate(force=True)
                        if message.done is not None:
                            message.done.set()
                except Exception as exc:  # noqa: BLE001 - recorder failures must not kill inference
                    LOGGER.warning("clip recorder operation failed: %s", exc)
                    with self._lock:
                        self.stats.failed_writes += 1
                    if isinstance(message, _RotateMessage) and message.done is not None:
                        message.done.set()
                finally:
                    self._queue.task_done()
        finally:
            self._shutdown()

    def _handle_frame(self, message: _FrameMessage) -> None:
        state = self._state(message.camera_id)
        frame = message.frame
        image = frame.image
        height, width = image.shape[:2]
        frame_size = (width, height)
        if state.frame_size is None:
            state.frame_size = frame_size
        # Track timing unconditionally: an event must still produce a clip even
        # when the segment writer cannot open (headless build without a working
        # codec). Otherwise last_time_sec stays None and _handle_event drops the
        # event silently, losing the evidence clip entirely.
        state.last_time_sec = frame.time_sec
        try:
            if state.current is None:
                state.current = self._open_segment(state, frame.time_sec, frame_size)
            elif frame.time_sec - state.current.start_time_sec >= self.config.segment_seconds:
                closed = self._close_segment(state)
                if closed is not None:
                    self._append_segment_to_active_clips(closed)
                state.current = self._open_segment(state, frame.time_sec, frame_size)
            if state.current is not None:
                state.current.writer.write(_as_bgr(image))
                state.current.frame_count += 1
                state.current.end_time_sec = frame.time_sec
        except Exception as exc:  # noqa: BLE001 - encoder loss must not drop events
            LOGGER.warning("clip segment write failed (%s); continuing without video", exc)
            state.current = None
            with self._lock:
                self.stats.failed_writes += 1
        self._finalize_ready_clips()
        self._prune_segment_ring(state)

    def _handle_event(self, message: _EventMessage) -> None:
        state = self._states.get(message.camera_id)
        if state is None or state.last_time_sec is None or state.frame_size is None:
            self._clear_clip_id(message.camera_id, message.clip_id)
            if not message.allow_new_clip:
                with self._lock:
                    self.stats.attach_missed_events += 1
            return
        clip = self._active_clip(message.camera_id, message.clip_id)
        if clip is None:
            clip = self._active_clip_for_camera(message.camera_id)
        if clip is None and not message.allow_new_clip:
            self._clear_clip_id(message.camera_id, message.clip_id)
            with self._lock:
                self.stats.attach_missed_events += 1
            return
        if clip is not None:
            if message.event_ref not in clip.event_refs:
                clip.event_refs.append(message.event_ref)
            self._finalize_ready_clips()
            return
        try:
            clip = self._open_clip(message, state, state.last_time_sec, state.frame_size)
        except _ClipIdCollisionError:
            # External interference reused the reserved id: discard the event
            # (counted via _record_clip_id_collision) and never touch the
            # pre-existing artifact. The returned id maps to nothing.
            self._clear_clip_id(message.camera_id, message.clip_id)
            return
        except Exception:
            self._clear_clip_id(message.camera_id, message.clip_id)
            shutil.rmtree(
                self.config.store_dir / "clips" / ".staging" / message.clip_id,
                ignore_errors=True,
            )
            raise
        self._active_clips.append(clip)
        self._update_active_clip_count()
        for segment in list(state.closed_segments):
            self._append_segment_to_clip(segment, clip)
        self._finalize_ready_clips()

    def _active_clip(self, camera_id: str, clip_id: str) -> _ActiveClip | None:
        for clip in self._active_clips:
            if clip.camera_id == camera_id and clip.clip_id == clip_id:
                return clip
        return None

    def _active_clip_for_camera(self, camera_id: str) -> _ActiveClip | None:
        for clip in self._active_clips:
            if clip.camera_id == camera_id:
                return clip
        return None

    def _clear_clip_id(self, camera_id: str, clip_id: str) -> None:
        with self._lock:
            if self._clip_ids_by_camera.get(camera_id) == clip_id:
                self._clip_ids_by_camera.pop(camera_id, None)

    def _reserve_clip_id(self, camera_id: str) -> str:
        """Reserve a collision-free clip id at admission time.

        The returned id is the artifact's final identity; the actor never
        substitutes another id. Bounded retries guard against the (uuid4)
        astronomically-unlikely case that a generated id already has final or
        staging output on disk.
        """
        clips_dir = self.config.store_dir / "clips"
        staging_root = clips_dir / ".staging"
        clip_id = _clip_id(camera_id)
        for _ in range(8):
            if not (clips_dir / clip_id).exists() and not (staging_root / clip_id).exists():
                break
            # Caller (on_event) already holds self._lock; do not re-acquire it.
            self.stats.clip_id_collisions += 1
            clip_id = _clip_id(camera_id)
        return clip_id

    def _state(self, camera_id: str) -> _CameraState:
        state = self._states.get(camera_id)
        if state is None:
            segment_dir = self.config.store_dir / "segments" / _safe_name(camera_id)
            segment_dir.mkdir(parents=True, exist_ok=True)
            state = _CameraState(camera_id=camera_id, segment_dir=segment_dir)
            self._states[camera_id] = state
        return state

    def _open_segment(
        self, state: _CameraState, start_time_sec: float, frame_size: tuple[int, int]
    ) -> _OpenSegment:
        segment_stem = f"seg-{_utc_compact()}-{uuid.uuid4().hex[:8]}"
        path, writer, codec = self._open_writer_for_camera(
            state.camera_id, state.segment_dir, segment_stem, frame_size
        )
        return _OpenSegment(
            camera_id=state.camera_id,
            path=path,
            start_time_sec=start_time_sec,
            end_time_sec=start_time_sec,
            started_at=datetime.now(UTC),
            writer=writer,
            codec=codec,
        )

    def _close_segment(self, state: _CameraState) -> _Segment | None:
        current = state.current
        if current is None:
            return None
        current.writer.release()
        state.current = None
        if current.frame_count <= 0:
            _unlink_missing_ok(current.path)
            return None
        segment = _Segment(
            camera_id=current.camera_id,
            path=current.path,
            start_time_sec=current.start_time_sec,
            end_time_sec=current.end_time_sec,
            started_at=current.started_at,
            frame_count=current.frame_count,
        )
        state.closed_segments.append(segment)
        return segment

    def _open_clip(
        self,
        message: _EventMessage,
        state: _CameraState,
        event_time_sec: float,
        frame_size: tuple[int, int],
    ) -> _ActiveClip:
        # Identity contract: the clip_id returned by on_event() IS the clip_id of
        # the produced artifact. The id was reserved collision-free at admission
        # time (_reserve_clip_id); the actor never substitutes a different id.
        # Any residual collision (external interference) discards the event via
        # _ClipIdCollisionError instead of overwriting or silently renaming.
        clip_id = message.clip_id
        clips_dir = self.config.store_dir / "clips"
        staging_root = clips_dir / ".staging"
        final_dir = clips_dir / clip_id
        staging_dir = staging_root / clip_id
        if final_dir.exists() or staging_dir.exists():
            self._record_clip_id_collision()
            raise _ClipIdCollisionError(clip_id)
        try:
            staging_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            self._record_clip_id_collision()
            raise _ClipIdCollisionError(clip_id) from None
        if final_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            self._record_clip_id_collision()
            raise _ClipIdCollisionError(clip_id)
        writer: _FfmpegWriter | None = None
        tmp_video_path = staging_dir / "clip.tmp.mp4"
        final_video_path: Path | None = staging_dir / "clip.mp4"
        codec: str | None = None
        video_error: str | None = None
        try:
            tmp_video_path, writer, codec = self._open_writer_for_camera(
                message.camera_id, staging_dir, "clip.tmp", frame_size
            )
            final_video_path = staging_dir / f"clip{tmp_video_path.suffix}"
        except Exception as exc:  # noqa: BLE001 - manifest must survive encoder loss
            video_error = str(exc)
            final_video_path = None
        start_time_sec = event_time_sec
        started_at = datetime.now(UTC)
        pre_start = event_time_sec - self.config.pre_event_seconds
        candidates = [
            segment for segment in state.closed_segments if segment.end_time_sec >= pre_start
        ]
        if candidates:
            first = min(candidates, key=lambda segment: segment.start_time_sec)
            start_time_sec = first.start_time_sec
            started_at = first.started_at
        return _ActiveClip(
            camera_id=message.camera_id,
            event_ref=message.event_ref,
            event_type=message.event_type,
            clip_id=clip_id,
            event_time_sec=event_time_sec,
            cutoff_time_sec=event_time_sec + self.config.post_event_seconds,
            staging_dir=staging_dir,
            final_dir=final_dir,
            tmp_video_path=tmp_video_path,
            final_video_path=final_video_path,
            manifest_path=staging_dir / "manifest.json",
            writer=writer,
            codec=codec,
            opened_monotonic=time.monotonic(),
            video_error=video_error,
            started_at=started_at,
            start_time_sec=start_time_sec,
            last_time_sec=start_time_sec,
            event_refs=[message.event_ref],
        )

    def _append_segment_to_active_clips(self, segment: _Segment) -> None:
        for clip in list(self._active_clips):
            self._append_segment_to_clip(segment, clip)
        self._finalize_ready_clips()

    def _append_segment_to_clip(self, segment: _Segment, clip: _ActiveClip) -> None:
        if segment.camera_id != clip.camera_id or segment.path in clip.appended_paths:
            return
        window_start = clip.event_time_sec - self.config.pre_event_seconds
        if segment.end_time_sec < window_start or segment.start_time_sec > clip.cutoff_time_sec:
            return
        if clip.writer is not None:
            try:
                frames = _append_video(segment.path, clip.writer)
            except Exception as exc:  # noqa: BLE001 - finalize manifest without video
                clip.writer.release()
                clip.writer = None
                clip.video_error = str(exc)
            else:
                clip.frame_count += frames
        clip.appended_paths.add(segment.path)
        clip.last_time_sec = max(
            clip.last_time_sec, min(segment.end_time_sec, clip.cutoff_time_sec)
        )
        if segment.start_time_sec < clip.start_time_sec:
            clip.start_time_sec = segment.start_time_sec
            clip.started_at = segment.started_at

    def _finalize_ready_clips(self, *, force: bool = False) -> None:
        remaining: list[_ActiveClip] = []
        finalized = False
        now_monotonic = time.monotonic()
        for clip in self._active_clips:
            timed_out = now_monotonic - clip.opened_monotonic > (
                self._active_clip_deadline_sec() + clip.force_finalize_extension_sec
            )
            ready = force or clip.last_time_sec >= clip.cutoff_time_sec or timed_out
            if ready:
                self._clear_clip_id(clip.camera_id, clip.clip_id)
                self._finalize_clip(clip, forced=force or timed_out)
                finalized = True
            else:
                remaining.append(clip)
        self._active_clips = remaining
        self._update_active_clip_count()
        if finalized:
            self._rotate()

    def _finalize_clip(self, clip: _ActiveClip, *, forced: bool = False) -> None:
        writer = clip.writer
        reason_code = (
            EvidenceReasonCode.ENCODER_FAILED if clip.video_error is not None else None
        )
        if writer is not None:
            writer.release()
            if writer.failed:
                reason_code = EvidenceReasonCode.ENCODER_FAILED
        video_available = writer is not None and clip.frame_count > 0 and not writer.failed
        clip.final_dir.mkdir(parents=True, exist_ok=False)
        _fsync_directory(clip.final_dir.parent)
        video_path: str | None = None
        destination: Path | None = None
        if video_available and clip.final_video_path is not None:
            destination = clip.final_dir / clip.final_video_path.name
            try:
                os.replace(clip.tmp_video_path, destination)
                _fsync_file(destination)
                _fsync_directory(clip.final_dir)
            except Exception:  # noqa: BLE001 - durable typed failure survives platform errors
                video_available = False
                reason_code = EvidenceReasonCode.FINALIZE_FAILED
        finalized_at = datetime.now(UTC)
        duration_s = round(max(0.0, clip.last_time_sec - clip.start_time_sec), 3)
        capture_end_at = min(
            clip.started_at + timedelta(seconds=duration_s),
            finalized_at,
        )
        capture_start_at = capture_end_at - timedelta(seconds=duration_s)
        event_refs = tuple(EdgeEventId(event_ref) for event_ref in clip.event_refs)
        manifest: ClipManifest | None = None
        if video_available and destination is not None:
            try:
                manifest = finalize_ready_manifest(
                    video_path=destination,
                    clip_id=ClipId(clip.clip_id),
                    camera_id=clip.camera_id,
                    event_refs=event_refs,
                    clip_start_at=capture_start_at,
                    clip_end_at=capture_end_at,
                    finalized_at=finalized_at,
                    ffprobe_bin=_ffprobe_bin(),
                )
                video_path = f"clips/{clip.clip_id}/{destination.name}"
            except ClipEvidenceError as exc:
                video_available = False
                reason_code = exc.reason_code
        if not video_available:
            if reason_code is None:
                reason_code = (
                    EvidenceReasonCode.NO_FRAMES
                    if clip.frame_count <= 0
                    else EvidenceReasonCode.ENCODER_FAILED
                )
            manifest = unavailable_manifest(
                clip_id=ClipId(clip.clip_id),
                camera_id=clip.camera_id,
                event_refs=event_refs,
                clip_start_at=capture_start_at,
                clip_end_at=capture_end_at,
                finalized_at=finalized_at,
                reason_code=reason_code,
            )
        if manifest is None:
            raise AssertionError("finalization must produce a manifest")
        payload = manifest.model_dump(mode="json")
        payload.update(
            {
                "event_ref": clip.event_ref,
                "started_at": _utc_iso(clip.started_at),
                "duration_s": duration_s,
                "encoder": clip.codec,
                "path": video_path,
                "finalized": True,
                "video_available": video_available,
            }
        )
        if clip.event_type is not None:
            payload["event_type"] = clip.event_type
        _atomic_write_json(clip.final_dir / "manifest.json", payload)
        shutil.rmtree(clip.staging_dir)
        _fsync_directory(clip.staging_dir.parent)
        with self._lock:
            self.stats.finalized_clips += 1
            if not video_available:
                self.stats.video_unavailable_clips += 1
            if forced:
                self.stats.forced_finalized += 1
        if self._on_clip_finalized is not None:
            try:
                self._on_clip_finalized(ClipId(clip.clip_id))
            except Exception as exc:  # noqa: BLE001 - durable artifact survives callback loss
                LOGGER.warning("clip finalize notification failed: %s", exc)

    def _record_clip_id_collision(self) -> None:
        with self._lock:
            self.stats.clip_id_collisions += 1
    def _open_writer_for_camera(
        self,
        camera_id: str,
        directory: Path,
        stem: str,
        frame_size: tuple[int, int],
    ) -> tuple[Path, _FfmpegWriter, str]:
        encoder = _resolve_encoder()
        if encoder == NVENC_ENCODER and min(frame_size) < NVENC_MIN_PROBE_DIMENSION:
            # Keep synthetic/thumbnail feeds usable: the NVENC probe documents
            # that sub-256 frames cannot be encoded by this hardware path.
            encoder = SOFTWARE_ENCODER
        path = directory / f"{stem}.mp4"
        writer = _open_writer(
            path, frame_size, self._fps_by_camera.get(camera_id, self.config.fps), encoder
        )
        self._codec_by_camera[camera_id] = _CodecSpec(encoder, ".mp4")
        return path, writer, encoder



    def _close_all_open_segments(self) -> None:
        for state in self._states.values():
            closed = self._close_segment(state)
            if closed is not None:
                self._append_segment_to_active_clips(closed)
    def _shutdown(self) -> None:
        try:
            self._close_all_open_segments()
            self._finalize_ready_clips(force=True)
            self._rotate(force=True)
        except Exception as exc:  # noqa: BLE001 - shutdown must release every writer
            LOGGER.warning("clip recorder shutdown finalize failed: %s", exc)
            with self._lock:
                self.stats.failed_writes += 1
        finally:
            for state in self._states.values():
                if state.current is not None:
                    writer = state.current.writer
                    state.current = None
                    self._release_writer_safely(writer)
            for clip in self._active_clips:
                if clip.writer is not None:
                    writer = clip.writer
                    clip.writer = None
                    self._release_writer_safely(writer)
            self._active_clips.clear()
            with self._lock:
                self._clip_ids_by_camera.clear()
                self.stats.active_clips = 0
    def _release_writer_safely(self, writer: _FfmpegWriter) -> None:
        try:
            writer.release()
        except Exception as exc:  # noqa: BLE001 - every writer must get a release attempt
            LOGGER.warning("clip recorder writer release failed: %s", exc)
            with self._lock:
                self.stats.failed_writes += 1

    def _prune_segment_ring(self, state: _CameraState) -> None:
        if state.last_time_sec is None:
            return
        keep_after = state.last_time_sec - self.config.pre_event_seconds
        kept: list[_Segment] = []
        for segment in state.closed_segments:
            if segment.end_time_sec >= keep_after:
                kept.append(segment)
            else:
                _unlink_missing_ok(segment.path)
        state.closed_segments = kept
    def _update_active_clip_count(self) -> None:
        with self._lock:
            self.stats.active_clips = len(self._active_clips)

    def _active_clip_deadline_sec(self) -> float:
        return max(
            0.0,
            self.config.pre_event_seconds
            + self.config.post_event_seconds
            + (self.config.segment_seconds * 2)
            + self.config.finalize_grace_seconds,
        )

    def _sweep_stale_staging(self) -> None:
        staging_root = self.config.store_dir / "clips" / ".staging"
        cutoff = time.time() - max(0.0, self.config.stale_staging_seconds)
        cleaned = 0
        for staging_dir in staging_root.iterdir():
            try:
                if self._retention.is_held(staging_dir.name):
                    continue
                if not staging_dir.is_dir() or staging_dir.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(staging_dir)
                cleaned += 1
            except OSError as exc:
                LOGGER.warning("clip staging cleanup failed for %s: %s", staging_dir.name, exc)
        if cleaned:
            with self._lock:
                self.stats.stale_staging_cleaned += cleaned


    def _rotate(self, *, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        if (
            not force
            and self._last_rotate_monotonic is not None
            and now_monotonic - self._last_rotate_monotonic
            < self.config.rotate_min_interval_seconds
        ):
            return
        self._last_rotate_monotonic = now_monotonic
        clips = _finalized_clips(self.config.store_dir)
        now = datetime.now(UTC)
        report = self._retention.rotate(
            (
                PurgeCandidate(
                    clip_id=clip_dir.name,
                    clip_dir=clip_dir,
                    finalized_at=started_at,
                )
                for started_at, clip_dir in clips
            ),
            retention_cutoff=now - timedelta(days=self.config.retention_days),
            disk_high_watermark=self.config.disk_high_watermark,
        )
        with self._lock:
            self.stats.held_clips = len(report.held_clip_ids)
            self.stats.purge_failures += len(report.failure_clip_ids)
            self.stats.recording_suspended = report.pressure_blocked


class ClipRecorderProtocol(Protocol):
    def on_frame(self, camera_id: str, frame: Frame) -> bool: ...
    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None: ...


def _ffprobe_bin() -> str:
    binary = _ffmpeg_bin()
    if binary.endswith("ffmpeg"):
        return f"{binary[:-len('ffmpeg')]}ffprobe"
    return "ffprobe"


def _ffmpeg_bin() -> str:
    return _env_str(CLIP_FFMPEG_BIN_ENV, DEFAULT_FFMPEG_BIN)


def _ffmpeg_encode_args(
    ffmpeg_bin: str, path: Path, frame_size: tuple[int, int], fps: float, encoder: str
) -> list[str]:
    width, height = frame_size
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{max(fps, 1.0):g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]


def _probe_nvenc(ffmpeg_bin: str) -> bool:
    """Prove NVENC actually encodes, not merely that the encoder is listed.

    Encoder-list presence does not guarantee runtime success: libnvidia-encode
    may be absent (missing the ``video`` driver capability), the driver may
    mismatch, or NVENC sessions may be exhausted. The 256x256 probe is required
    because NVENC rejects the former 64x64 input as below its minimum resolution.
    A one-frame synthetic encode is the only reliable readiness signal.
    """
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:r=25:d=0.08",
        "-frames:v",
        "1",
        "-c:v",
        NVENC_ENCODER,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _resolve_encoder() -> str:
    global _resolved_encoder
    with _encoder_lock:
        if _resolved_encoder is not None:
            return _resolved_encoder
        ffmpeg_bin = _ffmpeg_bin()
        encoder = NVENC_ENCODER if _probe_nvenc(ffmpeg_bin) else SOFTWARE_ENCODER
        LOGGER.info("clip recorder using ffmpeg encoder: %s", encoder)
        _resolved_encoder = encoder
        return encoder


class _FfmpegWriter:
    """Duck-typed replacement for cv2.VideoWriter backed by an ffmpeg process.

    Exposes only the ``write``/``release`` surface the recorder uses. Frames are
    piped in as raw bgr24 bytes and muxed to an H.264 + yuv420p + faststart MP4.
    Any failure is captured on ``failed``/``error`` instead of raised, so a lost
    encoder degrades a clip to ``video_available=false`` rather than crashing the
    recorder thread (matching the existing cv2 error-handling seams).
    """

    def __init__(
        self,
        path: Path,
        frame_size: tuple[int, int],
        fps: float,
        encoder: str,
        ffmpeg_bin: str,
    ) -> None:
        self.path = path
        self.encoder = encoder
        self.failed = False
        self.error: str | None = None
        self._frame_size = frame_size
        path.parent.mkdir(parents=True, exist_ok=True)
        args = _ffmpeg_encode_args(ffmpeg_bin, path, frame_size, fps, encoder)
        try:
            self._proc: subprocess.Popen[bytes] | None = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._proc = None
            self.failed = True
            self.error = str(exc)
            raise RuntimeError(f"failed to start ffmpeg encoder: {exc}") from exc

    def write(self, frame) -> None:  # noqa: ANN001 - numpy dtype carried by Frame
        proc = self._proc
        if proc is None or proc.stdin is None or self.failed:
            return
        height, width = frame.shape[:2]
        if (width, height) != self._frame_size:
            # Rawvideo has no per-frame size; a mismatched frame would desync the
            # whole stream, so drop it rather than corrupt every later frame.
            return
        try:
            proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError, OSError) as exc:
            self.failed = True
            self.error = str(exc)

    def release(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.returncode not in (0, None) and not self.failed:
            self.failed = True
            self.error = f"ffmpeg exited with code {proc.returncode}"


def _open_writer(
    path: Path, frame_size: tuple[int, int], fps: float, encoder: str
) -> _FfmpegWriter:
    return _FfmpegWriter(path, frame_size, fps, encoder, _ffmpeg_bin())


def _append_video(path: Path, writer: _FfmpegWriter) -> int:
    capture = cv2.VideoCapture(str(path))
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
    return frames


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_file(path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finalized_clips(store_dir: Path) -> list[tuple[datetime, Path]]:
    root = store_dir / "clips"
    if not root.exists():
        return []
    clips: list[tuple[datetime, Path]] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("finalized") is not True:
            continue
        finalized_at = _parse_utc(str(payload.get("finalized_at", "")))
        if finalized_at is None:
            try:
                finalized_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
            except OSError:
                continue
        clips.append((finalized_at, manifest_path.parent))
    return clips


def _parse_utc(value: str) -> datetime | None:
    if value == "":
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _as_bgr(image):  # noqa: ANN001 - numpy dtype is already carried by Frame
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _clip_id(camera_id: str) -> str:
    return f"{_safe_name(camera_id)}-{_utc_compact()}-{uuid.uuid4().hex[:12]}"


def _safe_name(value: str) -> str:
    return (
        "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
        or "camera"
    )


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _unlink_missing_ok(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return default if value == "" else value


def _env_retention_days() -> int:
    raw = os.environ.get(CLIP_STORE_RETENTION_DAYS_ENV, os.environ.get(CLIP_RETENTION_DAYS_ENV, ""))
    if raw.strip() == "":
        return DEFAULT_RETENTION_DAYS
    return max(MIN_RETENTION_DAYS, int(raw))


def _env_disk_high_watermark() -> float:
    raw = os.environ.get(
        CLIP_STORE_MAX_USAGE_ENV,
        os.environ.get(CLIP_DISK_HIGH_WATERMARK_ENV, ""),
    )
    if raw.strip() == "":
        return DEFAULT_DISK_HIGH_WATERMARK
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    return min(max(value, 0.01), 0.99)


__all__ = [
    "CLIP_STORE_DIR_ENV",
    "ClipRecorder",
    "ClipRecorderConfig",
    "ClipRecorderProtocol",
    "ClipRecorderStats",
]
