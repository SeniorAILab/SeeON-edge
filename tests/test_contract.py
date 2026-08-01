from __future__ import annotations

from contracts import DetectionResult


def test_detection_result_empty_defaults() -> None:
    result = DetectionResult()

    assert result.boxes == ()
    assert result.labels == ()
    assert result.keypoints == ()
