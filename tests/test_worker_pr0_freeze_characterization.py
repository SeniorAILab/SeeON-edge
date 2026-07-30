"""PR-1 tagged runner-result guard tests for CameraWorker behavior.

These tests document the post-refactor behavior:

- ``_run_runner`` invokes the unified runner seam and returns an explicit
  tagged ``RunnerResult``; legacy method probe order is intentionally gone.
- ``_build_observation`` dispatches on ``RunnerResult.kind`` instead of
  sniffing raw result shapes.
- Two distinct failure domains: SOURCE failure (construction/iteration) ->
  ``_mark_source_failure`` -> ``DEGRADED`` status + ``camera.offline`` ops event;
  per-frame PROCESSING failure -> ``_mark_processing_failure`` ->
  ``frame.processing_error`` ops event and NOT marked offline (must be preserved
  through the PR-4a RTSP reconnect/liveness work).
- Event enrichment ``setdefault`` semantics and detector-result normalization.

Frozen invariant (not under test here, documented for reviewers): the
worker->ml-api relay paths in ``worker/edge_worker_config.py`` and the shared
``_RunnerBundle`` runner identity are load-bearing and must remain green under
``tests/test_worker_backend_ingest_contract.py`` and
``tests/test_worker_runner_sharing.py``.
"""

from __future__ import annotations

import pytest

from contracts.frame import Frame
from contracts.observation import BoundingBox, DetectionResult, FrameObservation
from contracts.runner import bed_result, detection_result, person_result, pose_result
from edge.runtime import camera_worker as cw
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.status_store import CameraStatus, StatusStore


def _frame(index: int = 0) -> Frame:
    return Frame(index=index, time_sec=0.0, image=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _run_runner unified seam
# --------------------------------------------------------------------------- #


class _RunRunner:
    def run(self, image: object):
        del image
        return person_result(((1, 2, 3, 4, 0.9),))


class _CallableRunner:
    def __call__(self, image: object):
        del image
        return person_result(((5, 6, 7, 8, 0.8),))


class _LegacyPredictOnly:
    def predict(self, image: object):
        del image
        return person_result(((9, 10, 11, 12, 0.7),))


class _NoInvocation:
    pass


def test_run_runner_invokes_unified_run_method() -> None:
    result = cw._run_runner(_RunRunner(), _frame())
    assert result.kind == "person"
    assert result.boxes == ((1, 2, 3, 4, 0.9),)


def test_run_runner_falls_back_to_callable() -> None:
    result = cw._run_runner(_CallableRunner(), _frame())
    assert result.kind == "person"
    assert result.boxes == ((5, 6, 7, 8, 0.8),)


def test_run_runner_does_not_probe_legacy_predict_methods() -> None:
    with pytest.raises(TypeError):
        cw._run_runner(_LegacyPredictOnly(), _frame())  # type: ignore[arg-type]


def test_run_runner_without_supported_invocation_raises() -> None:
    with pytest.raises(TypeError):
        cw._run_runner(_NoInvocation(), _frame())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _build_observation tagged result dispatch
# --------------------------------------------------------------------------- #


class _EmptySource:
    def __iter__(self):  # pragma: no cover - never iterated in these tests
        return iter(())


class _RecordingBuilder:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(
        self,
        *,
        detections: object = None,
        raw_boxes: object = None,
        poses: object = None,
        bed_boxes: object = None,
    ) -> FrameObservation:
        self.kwargs = {
            "detections": detections,
            "raw_boxes": raw_boxes,
            "poses": poses,
            "bed_boxes": bed_boxes,
        }
        return FrameObservation()


def _worker_with_builder(builder: _RecordingBuilder) -> CameraWorker:
    return CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=_EmptySource(),
        runners={},
        observation_builder=builder,
        scene_state=None,
        status_store=StatusStore(),
    )


def test_build_observation_detection_result_routes_to_detections() -> None:
    rec = _RecordingBuilder()
    worker = _worker_with_builder(rec)
    detection = DetectionResult(boxes=(BoundingBox(0, 0, 1, 1, 0.9),))

    worker._build_observation({"pose": detection_result(detection)}, frame_index=None)

    assert rec.kwargs is not None
    assert rec.kwargs["detections"] is detection
    assert rec.kwargs["poses"] is None
    assert rec.kwargs["raw_boxes"] is None


def test_build_observation_pose_result_routes_to_poses_and_boxes() -> None:
    rec = _RecordingBuilder()
    worker = _worker_with_builder(rec)
    poses = [[(1, 2, 0.9)]]
    boxes = [[1, 2, 3, 4]]

    worker._build_observation({"pose": pose_result(poses, boxes)}, frame_index=None)

    assert rec.kwargs is not None
    assert rec.kwargs["poses"] is poses
    assert rec.kwargs["raw_boxes"] is boxes
    assert rec.kwargs["detections"] is None


def test_build_observation_detection_result_overrides_and_clears_raw_boxes() -> None:
    rec = _RecordingBuilder()
    worker = _worker_with_builder(rec)
    poses = [[(1, 2, 0.9)]]
    boxes = [[1, 2, 3, 4]]
    person = DetectionResult(boxes=(BoundingBox(5, 5, 6, 6, 0.8),))

    worker._build_observation(
        {"pose": pose_result(poses, boxes), "person": detection_result(person)}, frame_index=None
    )

    assert rec.kwargs is not None
    assert rec.kwargs["detections"] is person
    assert rec.kwargs["raw_boxes"] is None
    # pose keypoints still flow through even when person overrides detections
    assert rec.kwargs["poses"] is poses


def test_build_observation_person_result_sets_raw_boxes() -> None:
    rec = _RecordingBuilder()
    worker = _worker_with_builder(rec)
    boxes = [[1, 2, 3, 4]]

    worker._build_observation({"person": person_result(boxes)}, frame_index=None)

    assert rec.kwargs is not None
    assert rec.kwargs["raw_boxes"] is boxes
    assert rec.kwargs["detections"] is None


def test_build_observation_bed_output_parsed_to_bounding_boxes() -> None:
    rec = _RecordingBuilder()
    worker = _worker_with_builder(rec)

    worker._build_observation({"bed": bed_result([[1, 2, 3, 4, 0.9]])}, frame_index=None)

    assert rec.kwargs is not None
    bed_boxes = rec.kwargs["bed_boxes"]
    assert bed_boxes == (BoundingBox(1, 2, 3, 4, 0.9),)




# --------------------------------------------------------------------------- #
# detector-result normalization + identity enrichment
# --------------------------------------------------------------------------- #


def test_events_from_detector_normalizes_variants() -> None:
    mapping = {"event_type": "fall"}
    assert list(cw._events_from_detector(None)) == []
    assert list(cw._events_from_detector(mapping)) == [mapping]
    assert list(cw._events_from_detector([mapping, mapping])) == [mapping, mapping]


def test_with_camera_identity_uses_setdefault() -> None:
    event = {"event_type": "fall", "camera_id": "keep-me"}
    enriched = cw._with_camera_identity(event, "cam-1", "facility-1", 1.5)
    assert enriched["camera_id"] == "keep-me"  # existing value preserved
    assert enriched["facility_id"] == "facility-1"
    assert enriched["time_sec"] == 1.5


# --------------------------------------------------------------------------- #
# failure domains: source vs per-frame processing
# --------------------------------------------------------------------------- #


class _RaisingIterSource:
    def __iter__(self):
        raise RuntimeError("open failed")


class _RaiseOnceThenStop:
    def __init__(self) -> None:
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        self._n += 1
        if self._n == 1:
            raise RuntimeError("read failed")
        raise StopIteration


class _OneFrameSource:
    def __iter__(self):
        return iter([_frame(0)])


def _raising_builder(**_kwargs: object) -> FrameObservation:
    raise ValueError("processing boom")


def test_source_construction_failure_marks_camera_offline() -> None:
    store = StatusStore()
    worker = CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=_RaisingIterSource(),
        runners={},
        status_store=store,
    )

    processed = worker.run(max_frames=5)

    assert processed == 0
    record = store.get_status("cam-1")
    assert record is not None
    assert record.status == CameraStatus.DEGRADED
    assert record.error_category == "RuntimeError"
    assert "camera.offline" in {event.event_type for event in store.ops_events()}


def test_source_iteration_failure_marks_camera_offline() -> None:
    store = StatusStore()
    worker = CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=_RaiseOnceThenStop(),
        runners={},
        status_store=store,
    )

    processed = worker.run(max_frames=5)

    assert processed == 0  # iteration failure does not count as a processed frame
    assert "camera.offline" in {event.event_type for event in store.ops_events()}


def test_processing_failure_is_not_marked_offline() -> None:
    store = StatusStore()
    worker = CameraWorker(
        camera_id="cam-1",
        facility_id="facility-1",
        frame_source=_OneFrameSource(),
        runners={},
        observation_builder=_raising_builder,
        status_store=store,
    )

    processed = worker.run(max_frames=1)

    assert processed == 1  # processed increments even after a per-frame failure
    event_types = {event.event_type for event in store.ops_events()}
    assert "frame.processing_error" in event_types
    assert "camera.offline" not in event_types
    record = store.get_status("cam-1")
    assert record is not None
    assert record.status == CameraStatus.READY  # NOT DEGRADED/OFFLINE
