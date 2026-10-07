from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from worker.adapters.deepstream.metadata import convert_frame
from worker.adapters.deepstream.sources import SourceTable
from worker.adapters.deepstream.tensor_rows import rows_from_tensor
from worker.adapters.media.rtsp_native_frame import grab_native_jpeg
from worker.interfaces.media_plane import (
    EarlyStopUnsupported,
    MediaPlane,
    MediaPlaneStatus,
    MetadataSlot,
    RecordingInfo,
    RecordingRefused,
    SnapshotUnavailable,
    SourceRosterFixed,
    SourceStatus,
)
from worker.types.metadata import SourceBinding

_PERCEPTION_HEARTBEAT_FRAMES = 900
_PROBE_CONSECUTIVE_FAILURE_THRESHOLD = 3

_EMPTY_POSE_ROWS: Final = np.zeros((0, 57), dtype=np.float32)
LOGGER = logging.getLogger(__name__)


class DeepStreamFlowStopTimeout(RuntimeError):
    ...


class FlowFactory(Protocol):
    def __call__(self, config: DeepStreamMediaPlaneConfig) -> _FlowHandle: ...


@dataclass(frozen=True, slots=True)
class DeepStreamMediaPlaneConfig:
    infer_config_path: str
    tracker_config_path: str
    tracker_library_path: str
    record_dir: Path
    record_cache_seconds: int
    frame_width: int
    frame_height: int
    transform_id: str = "deepstream-flow.v1"
    pipeline_name: str = "deepstream-media-plane"
    snapshot_branch_enabled: bool = True


@dataclass(slots=True)
class _Recording:
    session_id: int
    on_sealed: Callable[[RecordingInfo], None]
    seal_deadline: float
    sealed: bool = False


_SEAL_GRACE_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class _FlowHandle:
    flow: Any
    pipeline: Any
    record_config: Callable[..., Any]
    render_mode_discard: Any
    make_probe: Callable[[str, _Probe], Any]


def _default_flow_factory(config: DeepStreamMediaPlaneConfig) -> _FlowHandle:
    from pyservicemaker import (
        Flow,
        Pipeline,
        Probe,
        RecordConfig,
        RenderMode,
    )

    pipeline = Pipeline(config.pipeline_name)
    return _FlowHandle(
        flow=Flow(pipeline),
        pipeline=pipeline,
        record_config=RecordConfig,
        render_mode_discard=RenderMode.DISCARD,
        make_probe=lambda name, probe: Probe(name, _batch_operator(probe)),
    )


class _Probe:
    def __init__(self, plane: DeepStreamMediaPlane) -> None:
        self._plane = plane

    def handle_metadata(self, batch_meta: Any) -> None:
        for frame_meta in batch_meta.frame_items:
            self._plane.publish_frame(frame_meta)


def _batch_operator(probe: _Probe) -> Any:
    from pyservicemaker import BatchMetadataOperator

    class _Operator(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()

        def handle_metadata(self, batch_meta: Any) -> None:
            probe.handle_metadata(batch_meta)

    return _Operator()


class DeepStreamMediaPlane(MediaPlane):
    def __init__(
        self,
        config: DeepStreamMediaPlaneConfig,
        *,
        metadata_slot: MetadataSlot,
        flow_factory: FlowFactory = _default_flow_factory,
        snapshot_encoder: Callable[[str], bytes] | None = None,
        native_frame_grabber: Callable[[str], bytes] = grab_native_jpeg,
        worker_boot_id: str | None = None,
        child_instance_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._config = config
        self._slot = metadata_slot
        handle = flow_factory(config)
        self._handle = handle
        self._flow = handle.flow
        self._pipeline = handle.pipeline
        self._sources = SourceTable(
            worker_boot_id=worker_boot_id or str(uuid.uuid4()),
            child_instance_id=child_instance_id or str(uuid.uuid4()),
            transform_id=config.transform_id,
        )
        self._frames_without_pose_tensor: dict[str, int] = {}
        self._objects_observed: dict[str, int] = {}
        self._matched_tracks: dict[str, int] = {}
        self._accepting = True
        self._probe_failures: dict[str, int] = {}
        self._probe_fatal_error: str | None = None
        self._cameras_warned_without_pose_tensor: set[str] = set()
        self._commands: queue.Queue[
            tuple[Callable[[], Any], threading.Event | None, list[Any] | None]
        ] = queue.Queue()
        self._command_thread = threading.Thread(
            target=self._run_commands, name="deepstream-flow-commands", daemon=True
        )
        self._command_thread.start()
        self._live: set[str] = set()
        self._publish_sequence: dict[str, int] = {}
        self._unmapped_pads: set[int] = set()
        self._recordings: dict[str, _Recording] = {}
        self._abandoned_recordings = 0
        self._active_encode_sessions: set[int] = set()
        self._started = False
        self._flow_thread: threading.Thread | None = None
        self._flow_error: Exception | None = None
        self._flow_finished = threading.Event()
        self._probe = _Probe(self)
        self._snapshot_encoder = snapshot_encoder
        self._native_frame_grabber = native_frame_grabber
        self._snapshot_lock = threading.Lock()
        self._snapshot_dir = config.record_dir / ".snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        for path in self._snapshot_dir.glob("*.jpg"):
            path.unlink()

    def start(self) -> None:
        if self._started:
            return
        self._build_flow()
        self._started = True
        self._flow_thread = threading.Thread(
            target=self._run_flow, name="deepstream-flow", daemon=True
        )
        self._flow_thread.start()

    def _run_flow(self) -> None:
        try:
            self._flow()
        except Exception as error:  # noqa: BLE001
            self._flow_error = error
        finally:
            self._flow_finished.set()

    def stop(self) -> None:
        if self._started:
            self._accepting = False
            self._pipeline.stop()
            if self._flow_thread is not None:
                self._flow_thread.join(timeout=10.0)
                if self._flow_thread.is_alive():
                    raise DeepStreamFlowStopTimeout(
                        "DeepStream Flow did not stop within 10 seconds; "
                        "the pipeline may still be delivering buffers"
                    )
            self._started = False

    def published_frames(self, camera_id: str) -> int:
        return self._publish_sequence.get(camera_id, 0)

    def perception_counters(self, camera_id: str) -> tuple[int, int]:
        return (
            self._objects_observed.get(camera_id, 0),
            self._frames_without_pose_tensor.get(camera_id, 0),
        )

    def status(self) -> MediaPlaneStatus:
        error = self._flow_error
        return MediaPlaneStatus(
            fatal_error=(
                f"{type(error).__name__}: {error}" if error is not None else self._probe_fatal_error
            ),
            sources=tuple(
                SourceStatus(
                    camera_id, self._sources.binding(camera_id), camera_id in self._live, 0
                )
                for camera_id in self._sources.camera_ids()
            ),
            engine_identity="pyservicemaker-flow",
            nvenc_sessions_active=len(self._active_encode_sessions),
        )

    def camera_id_for_pad(self, pad_index: int) -> str | None:
        return self._sources.camera_id_for_pad(pad_index)

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        if self._started:
            raise SourceRosterFixed(
                f"cannot add {camera_id!r}: the Flow's sources are fixed once it runs; "
                "restart the worker to change the roster"
            )
        binding = self._sources.add(camera_id, uri)
        self._slot.register_source(binding)
        return binding

    def remove_source(self, camera_id: str) -> None:
        if self._started:
            raise SourceRosterFixed(
                f"cannot remove {camera_id!r}: the Flow's sources are fixed once it runs; "
                "restart the worker to change the roster"
            )
        self._slot.remove_source(camera_id)
        self._live.discard(camera_id)
        self._recordings.pop(camera_id, None)
        self._sources.remove(camera_id)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        del category
        binding = self._sources.rebuild(camera_id)
        self._slot.register_source(binding)
        self._live.discard(camera_id)
        return binding

    def snapshot(self, camera_id: str, *, draw_objects: bool = True) -> bytes:
        if camera_id not in self._sources.camera_ids():
            raise SnapshotUnavailable(f"unknown source has no OSD snapshot: {camera_id}")
        if self._snapshot_encoder is not None:
            return self._snapshot_encoder(camera_id)
        if not self._started:
            raise SnapshotUnavailable(
                f"source has not started its OSD snapshot branch: {camera_id}"
            )
        if camera_id not in self._live:
            raise SnapshotUnavailable(
                f"source has not published a frame for its OSD snapshot branch: {camera_id}"
            )
        with self._snapshot_lock:
            index = self._sources.pad_index(camera_id)
            try:
                for stale_path in self._snapshot_dir.glob("snapshot-*.jpg"):
                    stale_path.unlink()

                def select_source_configure_osd_and_open_valve() -> None:
                    self._pipeline["snapshot-tiler"].set({"show-source": index})
                    self._pipeline["snapshot-osd"].set(
                        {
                            "display-bbox": int(draw_objects),
                            "display-text": int(draw_objects),
                        }
                    )
                    self._pipeline["snapshot-valve"].set({"drop": False})

                self._call_on_pipeline(select_source_configure_osd_and_open_valve)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    candidates = tuple(self._snapshot_dir.glob("snapshot-*.jpg"))
                    if candidates:
                        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
                        jpeg = path.read_bytes()
                        if jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"):
                            return jpeg
                    time.sleep(0.01)
                raise SnapshotUnavailable(f"OSD snapshot branch timed out for source: {camera_id}")
            finally:

                def close_valve_and_reset_osd() -> None:
                    try:
                        self._pipeline["snapshot-valve"].set({"drop": True})
                    finally:
                        self._pipeline["snapshot-osd"].set({"display-bbox": 1, "display-text": 1})

                try:
                    self._call_on_pipeline(close_valve_and_reset_osd)
                finally:
                    for snapshot_path in self._snapshot_dir.glob("snapshot-*.jpg"):
                        snapshot_path.unlink(missing_ok=True)

    def native_snapshot(self, camera_id: str) -> bytes:
        if camera_id not in self._sources.camera_ids():
            raise SnapshotUnavailable(f"unknown source has no OSD snapshot: {camera_id}")
        return self._native_frame_grabber(self._sources.uri(camera_id))

    def start_recording(
        self,
        camera_id: str,
        *,
        lookback_sec: int,
        duration_sec: int,
        on_sealed: Callable[[RecordingInfo], None],
    ) -> int:
        if camera_id not in self._live:
            raise RecordingRefused(f"source has not published a frame: {camera_id}")
        existing = self._recordings.get(camera_id)
        if existing is not None:
            now = self._clock()
            if now < existing.seal_deadline:
                return existing.session_id
            self._abandoned_recordings += 1
            LOGGER.error(
                "smart record session %d on camera_id=%s never sealed within %.0fs; "
                "releasing the recording slot so new alerts can record again "
                "(abandoned_total=%d)",
                existing.session_id,
                camera_id,
                now - (existing.seal_deadline - duration_sec - _SEAL_GRACE_SECONDS),
                self._abandoned_recordings,
            )
            self._recordings.pop(camera_id, None)
        session_id = self._start_signal(camera_id, lookback_sec, duration_sec)
        self._recordings[camera_id] = _Recording(
            session_id,
            on_sealed,
            seal_deadline=self._clock() + duration_sec + _SEAL_GRACE_SECONDS,
        )
        return session_id

    @property
    def abandoned_recordings(self) -> int:
        return self._abandoned_recordings

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        recording = self._recordings.get(camera_id)
        if recording is None or recording.session_id != session_id:
            raise RecordingRefused(f"unknown recording session {session_id} for {camera_id}")
        raise EarlyStopUnsupported(
            f"pyservicemaker cannot stop session {session_id} on {camera_id} early; "
            "the recording seals at its start duration"
        )

    def _enqueue(self, command: Callable[[], Any]) -> None:
        self._commands.put((command, None, None))

    def _call_on_pipeline(self, command: Callable[[], Any]) -> Any:
        done = threading.Event()
        result: list[Any] = []
        self._commands.put((command, done, result))
        done.wait()
        value = result[0]
        if isinstance(value, BaseException):
            raise value
        return value

    def _run_commands(self) -> None:
        while True:
            command, done, result = self._commands.get()
            try:
                value: Any = command()
            except Exception as error:  # noqa: BLE001
                value = error
            if result is not None:
                result.append(value)
            if done is not None:
                done.set()

    def _build_flow(self) -> None:
        camera_ids = self._sources.camera_ids()
        uris = [self._sources.uri(camera_id) for camera_id in camera_ids]
        if not uris:
            return
        record = self._handle.record_config(
            recording_type="local",
            rec_cache=self._config.record_cache_seconds,
            rec_dir_path=str(self._config.record_dir),
        )
        flow = (
            self._flow.batch_capture(
                uris,
                record_config=record,
                width=self._config.frame_width,
                height=self._config.frame_height,
            )
            .infer(self._config.infer_config_path)
            .track(
                ll_config_file=self._config.tracker_config_path,
                ll_lib_file=self._config.tracker_library_path,
            )
        )
        flow.attach(what=self._handle.make_probe("media-plane-probe", self._probe))
        terminal_flow = self._build_snapshot_branch(flow)
        terminal_flow.render(mode=self._handle.render_mode_discard, enable_osd=False, sync=False)
        for camera_id in camera_ids:
            source = self._source_element(camera_id)
            source.set(
                {
                    "select-rtp-protocol": 4,
                    "latency": 200,
                    "init-rtsp-reconnect-interval": 5,
                    "rtsp-reconnect-interval": 5,
                }
            )

    def _build_snapshot_branch(self, flow: Any) -> Any:
        if not self._config.snapshot_branch_enabled:
            return flow
        fork = flow.fork()
        tee = fork._streams[0].originator  # noqa: SLF001
        tee_queue = "snapshot-tee-queue"
        valve = "snapshot-valve"
        tiler = "snapshot-tiler"
        convert = "snapshot-convert"
        osd = "snapshot-osd"
        post_osd_convert = "snapshot-post-osd-convert"
        caps = "snapshot-caps"
        encoder = "snapshot-encoder"
        sink = "snapshot-sink"
        self._pipeline.add("queue", tee_queue)
        self._pipeline.add("valve", valve, {"drop": True, "drop-mode": 2})
        self._pipeline.add(
            "nvmultistreamtiler",
            tiler,
            {
                "rows": 1,
                "columns": 1,
                "width": self._config.frame_width,
                "height": self._config.frame_height,
                "show-source": 0,
            },
        )
        self._pipeline.add("nvvideoconvert", convert, {"gpu-id": 0, "compute-hw": 1})
        self._pipeline.add(
            "nvdsosd",
            osd,
            {"gpu-id": 0, "display-bbox": 1, "display-text": 1},
        )
        self._pipeline.add("nvvideoconvert", post_osd_convert, {"gpu-id": 0, "compute-hw": 1})
        self._pipeline.add("capsfilter", caps, {"caps": "video/x-raw(memory:NVMM), format=I420"})
        self._pipeline.add("nvjpegenc", encoder)
        self._pipeline.add(
            "multifilesink",
            sink,
            {
                "location": str(self._snapshot_dir / "snapshot-%010d.jpg"),
                "next-file": 0,
                "max-files": 1,
                "async": False,
                "sync": False,
            },
        )
        self._pipeline.link(
            tee,
            tee_queue,
            valve,
            tiler,
            convert,
            osd,
            post_osd_convert,
            caps,
            encoder,
            sink,
        )
        return fork

    def _source_element(self, camera_id: str) -> Any:
        return self._pipeline[self._sources.source_name(camera_id)]

    def _start_signal(self, camera_id: str, lookback_sec: int, duration_sec: int) -> int:
        source_name = self._sources.source_name(camera_id)
        return int(
            self._call_on_pipeline(
                lambda: self._pipeline.start_recording(
                    source_name,
                    lookback_sec,
                    duration_sec,
                    lambda info, camera_id=camera_id: self._recording_done(camera_id, info),
                )
            )
        )

    def publish_frame(self, frame_meta: Any) -> None:
        if not self._accepting:
            return
        pad_index = int(frame_meta.pad_index)
        camera_id = self._sources.camera_id_for_pad(pad_index)
        if camera_id is None:
            if pad_index not in self._unmapped_pads:
                self._unmapped_pads.add(pad_index)
                LOGGER.warning(
                    "dropping frames from unmapped mux pad %d; known pads are %s",
                    pad_index,
                    sorted(self._sources.pad_index(name) for name in self._sources.camera_ids()),
                )
            return
        try:
            self._publish_frame(frame_meta, camera_id)
        except Exception:  # noqa: BLE001
            self._record_probe_failure(camera_id)

    def _record_probe_failure(self, camera_id: str) -> None:
        failures = self._probe_failures.get(camera_id, 0) + 1
        self._probe_failures[camera_id] = failures
        if failures == 1:
            LOGGER.exception(
                "dropping frame whose conversion failed inside the SDK probe "
                "camera_id=%s consecutive_failures=%d",
                camera_id,
                failures,
            )
        if failures == _PROBE_CONSECUTIVE_FAILURE_THRESHOLD:
            self._probe_fatal_error = (
                "DeepStream probe conversion failed consecutively "
                f"camera_id={camera_id} failures={failures}"
            )
            LOGGER.error("%s", self._probe_fatal_error)

    def _publish_frame(self, frame_meta: Any, camera_id: str) -> None:
        rows = None
        for tensor_meta in frame_meta.tensor_items:
            layer = tensor_meta.as_tensor_output().get_layers()["output0"]
            rows = rows_from_tensor(layer)
            break
        inference_tensor_present = rows is not None
        objects = sum(1 for _ in frame_meta.object_items)
        self._objects_observed[camera_id] = self._objects_observed.get(camera_id, 0) + objects
        if rows is None:
            self._frames_without_pose_tensor[camera_id] = (
                self._frames_without_pose_tensor.get(camera_id, 0) + 1
            )
            if camera_id not in self._cameras_warned_without_pose_tensor:
                self._cameras_warned_without_pose_tensor.add(camera_id)
                LOGGER.warning(
                    "no pose tensor metadata on the first frame from camera %s "
                    "(%d tracked objects on it); every track on such a frame is unmatched",
                    camera_id,
                    objects,
                )
            rows = _EMPTY_POSE_ROWS
        sequence = self._publish_sequence.get(camera_id, 0) + 1
        self._publish_sequence[camera_id] = sequence
        binding = self._sources.binding(camera_id)
        metadata = convert_frame(
            frame_meta,
            rows=rows,
            inference_tensor_present=inference_tensor_present,
            binding=binding,
            frame_w=self._config.frame_width,
            frame_h=self._config.frame_height,
            publish_sequence=sequence,
            boot_id=binding.worker_boot_id,
        )
        self._live.add(camera_id)
        self._matched_tracks[camera_id] = self._matched_tracks.get(camera_id, 0) + len(
            metadata.frame.association.track_ids
        )
        self._probe_failures.pop(camera_id, None)
        if sequence % _PERCEPTION_HEARTBEAT_FRAMES == 0:
            LOGGER.info(
                "perception heartbeat camera_id=%s frames=%d objects=%d matched_tracks=%d "
                "frames_without_pose_tensor=%d",
                camera_id,
                sequence,
                self._objects_observed[camera_id],
                self._matched_tracks[camera_id],
                self._frames_without_pose_tensor.get(camera_id, 0),
            )
        self._slot.publish(metadata)

    def _recording_done(self, camera_id: str, info: Any) -> None:
        recording = self._recordings.get(camera_id)
        if recording is None or recording.sealed:
            return
        path = str(Path(str(info.file_directory)) / str(info.file_name))
        try:
            recording.on_sealed(
                RecordingInfo(
                    session_id=int(info.session_id),
                    camera_id=camera_id,
                    path=path,
                    duration_ms=int(info.duration),
                    width=int(info.width),
                    height=int(info.height),
                )
            )
        except Exception as error:
            LOGGER.exception(
                "clip publication failed for camera_id=%s session=%s; media is retained at %s "
                "and will be republished (%s)",
                camera_id,
                info.session_id,
                path,
                type(error).__name__,
            )
            self._recordings.pop(camera_id, None)
            return
        recording.sealed = True
        self._recordings.pop(camera_id, None)


__all__ = [
    "DeepStreamFlowStopTimeout",
    "DeepStreamMediaPlane",
    "DeepStreamMediaPlaneConfig",
    "FlowFactory",
    "_FlowHandle",
]
