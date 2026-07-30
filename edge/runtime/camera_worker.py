from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from contracts.event import EventPayload, MutableEventPayload
from contracts.frame import Frame, FrameSource
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    DetectionResult,
    FrameObservation,
)
from contracts.runner import (
    BedBoxOutput,
    BedRunnerResult,
    BoxOutput,
    DetectionRunnerResult,
    PersonRunnerResult,
    PoseOutput,
    PoseRunnerResult,
    RunnerOutput,
    RunnerProtocol,
)
from contracts.tracker import TrackerProtocol
from edge.domains import DomainRegistration
from edge.domains.base import DomainDetector
from edge.domains.bed_exit.schema import DomainDebugSnapshot
from edge.evidence.event_identity import EventIdentityStore
from edge.evidence.snapshot_store import SnapshotStore
from edge.perception.domain_input import build_domain_input
from edge.perception.observation_builder import build_frame_observation
from edge.perception.scene_state import SceneState
from edge.perception.tracker import GreedyIouTracker
from edge.runtime.incident_manager import IncidentManager
from edge.runtime.overlay_renderer import OverlayRenderer
from edge.runtime.runtime_diagnostics import WorkerDiagnostics
from edge.runtime.scheduler import Scheduler
from edge.runtime.status_store import CameraStatus, StatusStore
from shared.events.schemas import build_audit_envelope


class EventSinkProtocol(Protocol):
    def emit(self, event: EventPayload) -> None: ...


class PublishEventSinkProtocol(Protocol):
    def publish(self, event: EventPayload) -> None: ...


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


class EvidenceStagerProtocol(Protocol):
    def stage(self, event: EventPayload) -> None: ...

    def complete(self, edge_event_id: str, clip_id: str | None) -> None: ...


class OverlaySinkProtocol(Protocol):
    def publish(
        self,
        camera_id: str,
        frame: Frame,
        observation: FrameObservation,
        debug_snapshots: tuple[DomainDebugSnapshot, ...],
    ) -> None: ...


ObservationBuilder = Callable[..., FrameObservation]


@dataclass(slots=True)
class CameraWorker:
    camera_id: str
    facility_id: str
    frame_source: FrameSource
    runners: Mapping[str, RunnerProtocol]
    observation_builder: ObservationBuilder = build_frame_observation
    scheduler: Scheduler = field(default_factory=Scheduler)
    domain_detectors: tuple[DomainDetector, ...] = ()
    event_sink: EventSinkProtocol | PublishEventSinkProtocol | None = None
    incident_manager: IncidentManager = field(default_factory=IncidentManager)
    status_store: StatusStore = field(default_factory=StatusStore)
    scene_state: SceneState | None = None
    overlay_sink: OverlaySinkProtocol | None = None
    tracker: TrackerProtocol = field(default_factory=GreedyIouTracker)
    snapshot_renderer: OverlayRenderer | None = None
    snapshot_store: SnapshotStore | None = field(default_factory=SnapshotStore)
    detector_version: str | None = None
    clip_recorder: ClipRecorderProtocol | None = None
    evidence_stager: EvidenceStagerProtocol | None = None
    diagnostics: WorkerDiagnostics | None = None
    _fps_timestamps: deque[float] = field(default_factory=deque, init=False, repr=False)
    clip_recording_min_interval_sec: float = 30.0
    event_identity_path: Path | None = None
    _last_clip_recorded_at_sec: float | None = field(default=None, init=False)
    _event_identity_store: EventIdentityStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scene_state is None:
            self.scene_state = SceneState(self.camera_id)
        if self.clip_recording_min_interval_sec < 0:
            raise ValueError("clip_recording_min_interval_sec must be >= 0")
        self._event_identity_store = EventIdentityStore(self.event_identity_path)

    def _bind_source_liveness_callbacks(self) -> None:
        bind = getattr(self.frame_source, "set_liveness_callbacks", None)
        if not callable(bind):
            return
        bind(
            on_reconnecting=lambda reason: self.status_store.record_camera_reconnecting(
                self.camera_id,
                self.facility_id,
                "rtsp_reconnecting",
                detail=reason,
            ),
            on_recovered=lambda reason: self.status_store.record_camera_recovered(
                self.camera_id,
                self.facility_id,
                detail=reason,
            ),
        )

    def run(self, *, max_frames: int | None = None) -> int:
        processed = 0
        self.status_store.set_status(self.camera_id, self.facility_id, CameraStatus.STARTING)
        if self.scene_state is not None:
            self.scene_state.reset_for_new_source("source_iterator_start")
        # Source construction is the camera/source boundary: failure soft-degrades.
        try:
            self._bind_source_liveness_callbacks()
            frame_iter = iter(self.frame_source)
        except Exception as exc:  # noqa: BLE001 - source construction soft-degrades worker
            self._mark_source_failure(exc)
            return processed
        self.status_store.set_status(self.camera_id, self.facility_id, CameraStatus.READY)
        while max_frames is None or processed < max_frames:
            try:
                frame = next(frame_iter)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001 - source iteration soft-degrades worker
                self._mark_source_failure(exc)
                if self.scene_state is not None:
                    self.scene_state.reset_for_new_source("source_failure")
                continue
            # Per-frame processing (runners/perception/domains/incident/sink) is a
            # distinct failure domain: it MUST NOT be misreported as camera.offline.
            try:
                self.process_frame(frame)
            except Exception as exc:  # noqa: BLE001 - processing error is distinct from source failure
                self._mark_processing_failure(exc)
            processed += 1
        return processed

    def _mark_processing_failure(self, exc: Exception) -> None:
        """Record a per-frame processing failure WITHOUT marking the camera offline."""
        category = exc.__class__.__name__
        self.status_store.record_ops_event(
            "frame.processing_error",
            self.camera_id,
            self.facility_id,
            category,
            detail=str(exc) or None,
        )

    def process_frame(self, frame: Frame) -> FrameObservation:
        self._record_measured_fps()
        self._record_clip_frame(frame)
        scheduled_tasks = self.scheduler.tasks_for_frame(frame.index)
        outputs = self._run_scheduled_runners(frame, scheduled_tasks)
        observation, _ = self._build_observation(
            outputs,
            frame_index=frame.index,
            bed_scheduled="bed" in scheduled_tasks,
            bed_interval=self.scheduler.task_intervals.get("bed", 30),
        )
        observation = replace(observation, track_ids=self.tracker.update(observation.boxes))
        assert self.scene_state is not None
        domain_input = build_domain_input(
            observation,
            frame_width=frame.image.shape[1],
            frame_height=frame.image.shape[0],
            live_track_ids=tuple(self.tracker.live_ids),
            time_sec=frame.time_sec,
            frame_index=frame.index,
            scene_state=self.scene_state,
            bed_scheduled="bed" in scheduled_tasks,
            bed_interval=self.scheduler.task_intervals.get("bed", 30),
        )
        self._record_bed_debug(domain_input.bed_region)
        prepared_inputs: list[tuple[DomainDetector, DomainRegistration | None]] = []
        canonical_input = domain_input
        for detector in self.domain_detectors:
            registration = _domain_registration(detector)
            if registration is not None:
                canonical_input = registration.input_preparer(
                    detector,
                    registration.input_view,
                    canonical_input,
                )
            prepared_inputs.append((detector, registration))
        observation = canonical_input.observation
        self.scene_state.update(
            observation,
            track_ids=tuple(track_id for track_id in observation.track_ids if track_id is not None),
        )
        debug_snapshots: list[DomainDebugSnapshot] = []
        for detector, registration in prepared_inputs:
            detector_result = detector.update(canonical_input)
            debug_snapshot = _domain_debug_snapshot(registration, detector, frame.index)
            if debug_snapshot is not None:
                debug_snapshots.append(debug_snapshot)
            for event in _events_from_detector(detector_result):
                event = _with_camera_identity(
                    event,
                    self.camera_id,
                    self.facility_id,
                    frame.time_sec,
                )
                if self.incident_manager.admit(event, now_sec=time.monotonic()):
                    event = self._event_identity_store.enrich(
                        event,
                        self.facility_id,
                        self.camera_id,
                    )
                    if self.evidence_stager is not None:
                        self._attach_alert_metadata(
                            event,
                            detector,
                            registration,
                            frame,
                            observation,
                            tuple(debug_snapshots),
                        )
                        self.evidence_stager.stage(event)
                    event["clip_id"] = self._record_clip_event(event)
                    if self.evidence_stager is not None:
                        clip_id = event.get("clip_id")
                        self.evidence_stager.complete(
                            _event_ref(event), None if clip_id is None else str(clip_id)
                        )
                    else:
                        self._attach_alert_metadata(
                            event,
                            detector,
                            registration,
                            frame,
                            observation,
                            tuple(debug_snapshots),
                        )
                        self._emit(event)
        if self.overlay_sink is not None:
            self.overlay_sink.publish(
                self.camera_id,
                frame,
                observation,
                tuple(debug_snapshots),
            )
        return observation
    def _record_measured_fps(self) -> None:
        if self.diagnostics is None:
            return
        now = time.monotonic()
        timestamps = self._fps_timestamps
        timestamps.append(now)
        while timestamps and now - timestamps[0] > 10.0:
            timestamps.popleft()
        if len(timestamps) >= 2:
            elapsed = timestamps[-1] - timestamps[0]
            self.diagnostics.update_measured_fps(
                self.camera_id, None if elapsed <= 0 else (len(timestamps) - 1) / elapsed
            )
    def _attach_alert_metadata(
        self,
        event: MutableEventPayload,
        detector: DomainDetector,
        registration: DomainRegistration | None,
        frame: Frame,
        observation: FrameObservation,
        debug_snapshots: tuple[DomainDebugSnapshot, ...],
    ) -> None:
        if registration is None or _event_type(event) not in registration.audit_event_types:
            return
        try:
            metadata = (
                registration.audit_metadata_provider(detector, detector.audit_context)
                if registration.audit_metadata_provider is not None
                else detector.audit_metadata()
            )
            event["audit"] = build_audit_envelope(
                model_version=_str_or_none(metadata.get("model_version")),
                detector_version=self.detector_version,
                operating_threshold=_float_or_none(metadata.get("operating_threshold")),
            )
            if self.snapshot_renderer is not None:
                snapshot = self.snapshot_renderer.encode_jpeg_bounded(
                    frame, observation, debug_snapshots
                )
                if snapshot is not None:
                    event["snapshot_jpeg"] = snapshot
                    if self.snapshot_store is not None:
                        stored = self.snapshot_store.store(
                            snapshot,
                            snapshot_id=_event_ref(event),
                            captured_at=str(event["detected_at"]),
                            camera_id=self.camera_id,
                            edge_event_id=str(event.get("edge_event_id") or "") or None,
                        )
                        event["snapshot"] = {
                            "snapshot_id": stored.snapshot_id,
                            "path": stored.path,
                            "sha256": stored.sha256,
                            "size_bytes": stored.size_bytes,
                            "mime_type": stored.mime_type,
                            "captured_at": stored.captured_at,
                            "camera_id": stored.camera_id,
                            "edge_event_id": stored.edge_event_id,
                        }
        except Exception:  # noqa: BLE001 - audit/snapshot metadata must not block alert emit
            return


    def _run_scheduled_runners(
        self,
        frame: Frame,
        scheduled_tasks: tuple[str, ...] | None = None,
    ) -> dict[str, RunnerOutput]:
        outputs: dict[str, RunnerOutput] = {}
        tasks = (
            scheduled_tasks
            if scheduled_tasks is not None
            else self.scheduler.tasks_for_frame(frame.index)
        )
        for task in tasks:
            runner = self.runners.get(task)
            if runner is None:
                continue
            outputs[task] = _run_runner(runner, frame)
        return outputs

    def _build_observation(
        self,
        outputs: Mapping[str, RunnerOutput],
        *,
        frame_index: int | None = None,
        bed_scheduled: bool = False,
        bed_interval: int = 30,
    ) -> tuple[FrameObservation, BedRegionDebugSnapshot]:
        detections: DetectionResult | None = None
        poses: PoseOutput | None = None
        raw_boxes: BoxOutput | None = None
        bed_boxes: tuple[BoundingBox, ...] | None = None

        pose_output = outputs.get("pose")
        if isinstance(pose_output, DetectionRunnerResult):
            detections = pose_output.detections
        elif isinstance(pose_output, PoseRunnerResult):
            poses = pose_output.poses
            raw_boxes = pose_output.boxes

        person_output = outputs.get("person")
        if isinstance(person_output, DetectionRunnerResult):
            detections = person_output.detections
            raw_boxes = None
        elif isinstance(person_output, PersonRunnerResult):
            raw_boxes = person_output.boxes
        bed_output = outputs.get("bed")
        if isinstance(bed_output, BedRunnerResult):
            bed_boxes = _bed_boxes_from_output(bed_output.boxes)

        observation = self.observation_builder(
            detections=detections,
            raw_boxes=raw_boxes,
            poses=poses,
            bed_boxes=bed_boxes,
        )
        debug = BedRegionDebugSnapshot(
            source=BedRegionCacheState.FRESH if bed_boxes else BedRegionCacheState.EMPTY,
            age_frames=0 if bed_boxes else None,
        )
        return observation, debug

    def _record_bed_debug(self, debug: BedRegionDebugSnapshot) -> None:
        self.status_store.record_ops_event(
            "bed_roi.cache",
            self.camera_id,
            self.facility_id,
            debug.source,
            detail=(
                f"age={debug.age_frames};empty_cycles={debug.empty_cycles}"
                + (f";reset={debug.reset_reason}" if debug.reset_reason else "")
            ),
        )

    def _record_clip_frame(self, frame: Frame) -> None:
        if self.clip_recorder is None:
            return
        try:
            self.clip_recorder.on_frame(self.camera_id, frame)
        except Exception:  # noqa: BLE001 - recorder backpressure/failure must not block inference
            return

    def _record_clip_event(self, event: EventPayload) -> str | None:
        if self.clip_recorder is None:
            return None
        now_sec = time.monotonic()
        last_recorded_at_sec = self._last_clip_recorded_at_sec
        throttled = (
            last_recorded_at_sec is not None
            and now_sec - last_recorded_at_sec < self.clip_recording_min_interval_sec
        )
        try:
            clip_id = self.clip_recorder.on_event(
                self.camera_id,
                _event_ref(event),
                _event_type(event),
                allow_new_clip=not throttled,
            )
        except Exception:  # noqa: BLE001 - recorder finalize failure must not block alert emit
            return None
        if clip_id is not None and not throttled:
            self._last_clip_recorded_at_sec = now_sec
        return clip_id

    def _emit(self, event: EventPayload) -> None:
        if self.event_sink is None:
            return
        emit = getattr(self.event_sink, "emit", None)
        if callable(emit):
            emit(event)
            return
        publish = getattr(self.event_sink, "publish", None)
        if callable(publish):
            publish(event)

    def _mark_source_failure(self, exc: Exception) -> None:
        category = exc.__class__.__name__
        self.status_store.record_camera_reconnecting(
            self.camera_id,
            self.facility_id,
            category,
            detail=str(exc) or None,
        )


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _domain_registration(detector: DomainDetector) -> DomainRegistration | None:
    registration = getattr(detector, "registration", None)
    return registration if isinstance(registration, DomainRegistration) else None


def _domain_debug_snapshot(
    registration: DomainRegistration | None,
    detector: DomainDetector,
    frame_index: int,
) -> DomainDebugSnapshot | None:
    if registration is None or registration.debug_snapshot_adapter is None:
        return None
    return registration.debug_snapshot_adapter(detector, frame_index)


def _run_runner(runner: RunnerProtocol, frame: Frame) -> RunnerOutput:
    method = getattr(runner, "run", None)
    if callable(method):
        return method(frame.image)
    if callable(runner):
        return runner(frame.image)
    raise TypeError(f"runner {runner!r} has no supported invocation method")




def _bed_box_from_output(item: BedBoxOutput) -> BoundingBox:
    values = tuple(item)
    if len(values) == 6:
        x1, y1, x2, y2, confidence, polygon = values
        return BoundingBox(int(x1), int(y1), int(x2), int(y2), float(confidence), tuple(polygon))
    x1, y1, x2, y2, confidence = values[:5]
    return BoundingBox(int(x1), int(y1), int(x2), int(y2), float(confidence))


def _bed_boxes_from_output(output: Iterable[BedBoxOutput]) -> tuple[BoundingBox, ...]:
    return tuple(_bed_box_from_output(item) for item in output)


def _events_from_detector(
    result: EventPayload | Iterable[EventPayload] | None,
) -> Iterator[EventPayload]:
    if result is None:
        return iter(())
    if isinstance(result, Mapping):
        return iter((result,))
    if isinstance(result, Iterable) and not isinstance(result, str | bytes):
        return iter(result)
    return iter((result,))


def _event_ref(event: EventPayload) -> str:
    for key in ("edge_event_id", "event_id", "identity", "detected_at", "time_sec"):
        value = event.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(event.get("event_type", "event"))


def _event_type(event: EventPayload) -> str | None:
    value = event.get("event_type")
    if value is None or str(value) == "":
        return None
    return str(value)

def _with_camera_identity(
    event: EventPayload,
    camera_id: str,
    facility_id: str,
    time_sec: float,
) -> EventPayload:
    enriched: MutableEventPayload = dict(event)
    enriched.setdefault("camera_id", camera_id)
    enriched.setdefault("facility_id", facility_id)
    enriched.setdefault("time_sec", time_sec)
    return enriched
