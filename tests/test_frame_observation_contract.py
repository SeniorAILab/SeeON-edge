from __future__ import annotations

from contracts import FrameObservation as ExportedFrameObservation
from contracts.observation import BoundingBox, DetectionLabel, DetectionResult, FrameObservation


def test_frame_observation_round_trips_representative_detection_result() -> None:
    result = DetectionResult(
        boxes=(
            BoundingBox(1, 2, 30, 40, 0.91),
            BoundingBox(50, 60, 90, 120, 0.82, polygon=((50, 60), (90, 60), (90, 120))),
        ),
        labels=(
            DetectionLabel("NORMAL", 0.88, False),
            DetectionLabel("FALL", 0.93, True),
        ),
        keypoints=(
            tuple((idx, idx + 1, idx / 100.0) for idx in range(17)),
            tuple((idx + 20, idx + 21, (idx + 20) / 100.0) for idx in range(17)),
        ),
        bed_boxes=(BoundingBox(5, 6, 70, 80, 0.77),),
        bed_exit_statuses=("occupied", {"track_id": 2, "status": "exit"}),
    )

    observation = FrameObservation.from_detection_result(result)

    assert ExportedFrameObservation is FrameObservation
    assert observation.detections == (result.boxes, result.labels)
    assert observation.poses == result.keypoints
    assert observation.regions == (result.bed_boxes, result.bed_exit_statuses)
    assert observation.boxes == result.boxes
    assert observation.labels == result.labels
    assert observation.keypoints == result.keypoints
    assert observation.bed_boxes == result.bed_boxes
    assert observation.bed_exit_statuses == result.bed_exit_statuses
    assert observation.to_detection_result() == result
    assert FrameObservation.from_detection_result(observation.to_detection_result()) == observation


def test_frame_observation_round_trips_empty_detection_result() -> None:
    result = DetectionResult()

    observation = FrameObservation.from_detection_result(result)

    assert observation == FrameObservation()
    assert observation.to_detection_result() == result
