from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from contracts import (
    BoundingBox,
    DetectionLabel,
    DetectionResult,
    FrameObservation,
    ModelModule,
)
from edge.sources import Frame, VideoFileSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_video(path: Path, frame_count: int = 6, fps: float = 6.0) -> Path:
    """Write a minimal uncompressed AVI so VideoCapture can read it reliably."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (32, 24))
    for i in range(frame_count):
        frame = np.full((24, 32, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class _FakeModule:
    """Minimal ModelModule that returns a fixed FrameObservation."""

    def predict(self, frame: Frame) -> FrameObservation:
        box = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
        label = DetectionLabel(text="fall", confidence=0.9, is_fall=True)
        return FrameObservation(detections=((box,), (label,)))


# ---------------------------------------------------------------------------
# ModelModule / DetectionResult tests
# ---------------------------------------------------------------------------


def test_detection_result_empty_defaults() -> None:
    result = DetectionResult()

    assert result.boxes == ()
    assert result.labels == ()
    assert result.keypoints == ()


def test_detection_result_boxes_and_labels_populated() -> None:
    box = BoundingBox(x1=10, y1=20, x2=50, y2=80, confidence=0.75)
    label = DetectionLabel(text="fall", confidence=0.75, is_fall=True)
    result = DetectionResult(boxes=(box,), labels=(label,))

    assert len(result.boxes) == 1
    assert result.boxes[0].confidence == 0.75
    assert result.labels[0].is_fall is True


def test_fake_module_satisfies_model_module_protocol() -> None:
    module = _FakeModule()
    frame = Frame(
        index=0,
        time_sec=0.0,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
    )

    assert isinstance(module, ModelModule)
    result = module.predict(frame)

    assert isinstance(result, FrameObservation)
    assert len(result.boxes) == 1
    assert result.boxes[0].x2 == 10
    assert result.labels[0].text == "fall"
    assert result.labels[0].is_fall is True


def test_fake_module_over_video_file_source_yields_results(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    module = _FakeModule()

    results = [module.predict(frame) for frame in VideoFileSource(video_path)]

    assert len(results) == 4
    assert all(isinstance(r, FrameObservation) for r in results)
    assert all(len(r.boxes) == 1 for r in results)
