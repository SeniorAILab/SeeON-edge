"""The trained fall scorer: published features, physical gate, refused boot."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_vector_classifier import OrtVectorClassifier
from worker.domains.fall.geometry_features import (
    GEOMETRY_FEATURE_DIM,
    collapse_gate,
    geometry_features,
)
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.domains.fall.trained_scorer import TrainedGeometryFallScorer

_WIDTH, _HEIGHT = 640, 360
_ASPECT = _WIDTH / _HEIGHT


def _row(bbox, torso_deg: float, *, legs_down: bool = True) -> tuple[float, ...]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = math.sin(math.radians(torso_deg)) * 40.0
    dy = math.cos(math.radians(torso_deg)) * 40.0
    hip_y = cy + dy / 2
    ankle_y = hip_y + 4.0 if legs_down else y2 - 4.0
    keypoints = [(0.0, 0.0, 0.0)] * 17
    keypoints[5] = (cx - 5 - dx / 2, cy - dy / 2, 0.9)
    keypoints[6] = (cx + 5 - dx / 2, cy - dy / 2, 0.9)
    keypoints[11] = (cx - 5 + dx / 2, hip_y, 0.9)
    keypoints[12] = (cx + 5 + dx / 2, hip_y, 0.9)
    keypoints[15] = (cx - 5 + dx, ankle_y, 0.9)
    keypoints[16] = (cx + 5 + dx, ankle_y, 0.9)
    return pose_bbox56_row(keypoints, bbox, _WIDTH, _HEIGHT)


_UPRIGHT = _row((300.0, 100.0, 360.0, 300.0), 5.0, legs_down=False)
_ON_FLOOR = _row((250.0, 240.0, 410.0, 300.0), 65.0)
_BENDING = _row((270.0, 190.0, 400.0, 300.0), 70.0, legs_down=False)


def _window(*segments: tuple[tuple[float, ...], int]):
    rows: list[tuple[float, ...]] = []
    for row, count in segments:
        rows.extend([row] * count)
    return tuple(rows)


class _ConstantClassifier:
    feature_dim = GEOMETRY_FEATURE_DIM

    def __init__(self, value: float) -> None:
        self.value = value
        self.seen: list[Sequence[float]] = []

    def positive_probability(self, vector: Sequence[float]) -> float:
        self.seen.append(vector)
        return self.value


def test_features_are_the_published_width_and_zero_without_enough_rows() -> None:
    full = geometry_features(_window((_UPRIGHT, 12), (_ON_FLOOR, 18)), _ASPECT)
    assert len(full) == GEOMETRY_FEATURE_DIM
    assert any(value != 0.0 for value in full)
    assert geometry_features(_window((_UPRIGHT, 4)), _ASPECT) == [0.0] * GEOMETRY_FEATURE_DIM


def test_gate_admits_a_collapse_and_refuses_a_bend() -> None:
    assert collapse_gate(_window((_UPRIGHT, 12), (_ON_FLOOR, 18)), _ASPECT) is True
    assert collapse_gate(_window((_UPRIGHT, 12), (_BENDING, 18)), _ASPECT) is False


def test_scorer_serves_the_model_only_for_a_gated_window() -> None:
    classifier = _ConstantClassifier(0.9)
    scorer = TrainedGeometryFallScorer(classifier, frame_width=_WIDTH, frame_height=_HEIGHT)

    fall = scorer.predict(_window((_UPRIGHT, 12), (_ON_FLOOR, 18)))
    assert fall.fall_transition == pytest.approx(0.9)
    assert fall.fallen == 0.0
    assert len(classifier.seen) == 1
    assert len(classifier.seen[0]) == GEOMETRY_FEATURE_DIM

    bend = scorer.predict(_window((_UPRIGHT, 12), (_BENDING, 18)))
    assert bend.fall_transition == 0.0
    # A refused window must never reach the model.
    assert len(classifier.seen) == 1


def test_scorer_rejects_a_classifier_with_the_wrong_feature_width() -> None:
    class _Narrow:
        feature_dim = 3

        def positive_probability(self, vector: Sequence[float]) -> float:
            return 0.0

    with pytest.raises(ValueError, match="features"):
        TrainedGeometryFallScorer(_Narrow(), frame_width=_WIDTH, frame_height=_HEIGHT)


def test_missing_model_file_refuses_to_load(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        OrtVectorClassifier.from_model_path(
            tmp_path / "absent.onnx", feature_dim=GEOMETRY_FEATURE_DIM
        )


def test_model_with_the_wrong_input_width_refuses_to_load(tmp_path: Path) -> None:
    model = tmp_path / "wrong.onnx"
    model.write_bytes(b"onnx")

    class _Input:
        name = "X"
        shape = [None, 7]

    class _Output:
        name = "probabilities"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def get_outputs(self):
            return [_Output()]

        def run(self, output_names, input_feed):
            raise AssertionError("must not run")

    with pytest.raises(ModelLoadError, match="must be \\(batch, 14\\)"):
        OrtVectorClassifier.from_model_path(
            model,
            feature_dim=GEOMETRY_FEATURE_DIM,
            session_factory=lambda *_: _Session(),
        )
