"""Pose-geometry fall scorer: an upright-to-collapsed transition inside the window."""

from __future__ import annotations

import math

from worker.domains.fall.classifier_v2 import FALL_WINDOW_FRAMES
from worker.domains.fall.geometry_scorer import (
    PoseGeometryFallScorer,
    geometry_fall_transition,
)
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.interfaces.fall_model import FallV2Probabilities
from worker.types import FallModelInput

_WIDTH, _HEIGHT = 640, 360
_ASPECT = _WIDTH / _HEIGHT


def _row(
    bbox: tuple[float, float, float, float], torso_deg: float, *, legs_down: bool = True
) -> tuple[float, ...]:
    """A person box with shoulders and hips at ``torso_deg`` from vertical.

    ``legs_down`` places the ankles level with the hips (lying); otherwise the
    ankles sit at the box bottom below the hips (standing or bending).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    length = 40.0
    dx = math.sin(math.radians(torso_deg)) * length
    dy = math.cos(math.radians(torso_deg)) * length
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


_UPRIGHT = _row((300.0, 100.0, 360.0, 300.0), 5.0, legs_down=False)  # 200 tall x 60 wide
_ON_FLOOR = _row((250.0, 240.0, 410.0, 300.0), 65.0)  # 60 tall x 160 wide, legs level
_CROUCH = _row((300.0, 180.0, 380.0, 300.0), 18.0, legs_down=False)  # laptop crouch


def _seated_row() -> tuple[float, ...]:
    """Sitting on a chair: hips at mid-box, torso upright, feet on the floor below."""
    keypoints = [(0.0, 0.0, 0.0)] * 17
    keypoints[5], keypoints[6] = (330.0, 200.0, 0.9), (350.0, 200.0, 0.9)
    keypoints[11], keypoints[12] = (332.0, 240.0, 0.9), (348.0, 240.0, 0.9)
    keypoints[15], keypoints[16] = (334.0, 296.0, 0.9), (346.0, 296.0, 0.9)
    return pose_bbox56_row(keypoints, (300.0, 180.0, 380.0, 300.0), _WIDTH, _HEIGHT)


_SEATED = _seated_row()
# Caregiver bending over a bed: box collapses and torso turns horizontal, but
# the legs stay planted below the hips (room 207, 2026-09-15).
_BENDING = _row((270.0, 190.0, 400.0, 300.0), 70.0, legs_down=False)


def _window(*segments: tuple[tuple[float, ...], int]) -> FallModelInput:
    rows: list[tuple[float, ...]] = []
    for row, count in segments:
        rows.extend([row] * count)
    assert len(rows) == FALL_WINDOW_FRAMES
    return tuple(rows)


def test_upright_then_collapsed_scores_a_full_transition() -> None:
    window = _window((_UPRIGHT, 12), (_ON_FLOOR, 18))
    assert geometry_fall_transition(window, _ASPECT) == 1.0


def test_collapse_is_recognised_from_a_short_upright_prefix() -> None:
    # Upright evidence anywhere in the first half is enough: the window that
    # starts 1.3 s before the collapse still qualifies, giving the policy the
    # three consecutive ticks it needs.
    window = _window((_UPRIGHT, 5), (_ON_FLOOR, 25))
    assert geometry_fall_transition(window, _ASPECT) == 1.0


def test_stable_crouch_and_stable_lying_do_not_score() -> None:
    assert geometry_fall_transition(_window((_CROUCH, 30)), _ASPECT) == 0.0
    assert geometry_fall_transition(_window((_ON_FLOOR, 30)), _ASPECT) == 0.0


def test_standing_up_runs_the_transition_backwards_and_does_not_score() -> None:
    window = _window((_ON_FLOOR, 12), (_UPRIGHT, 18))
    assert geometry_fall_transition(window, _ASPECT) == 0.0


def test_sitting_down_keeps_an_upright_torso_and_does_not_score() -> None:
    window = _window((_UPRIGHT, 12), (_SEATED, 18))
    assert geometry_fall_transition(window, _ASPECT) == 0.0


def test_bending_over_a_bed_keeps_the_legs_planted_and_does_not_score() -> None:
    window = _window((_UPRIGHT, 12), (_BENDING, 18))
    assert geometry_fall_transition(window, _ASPECT) == 0.0


def test_invalid_rows_and_flat_input_score_nothing() -> None:
    zero = (0.0,) * 56
    assert geometry_fall_transition(_window((_UPRIGHT, 12), (zero, 18)), _ASPECT) == 0.0
    assert geometry_fall_transition((0.0,) * 56, _ASPECT) == 0.0


class _FlatModel:
    transition: float
    warmed: bool

    def __init__(self, transition: float) -> None:
        self.transition = transition
        self.warmed = False

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(1.0 - self.transition, self.transition, 0.0)

    def warmup(self) -> None:
        self.warmed = True


def test_scorer_lifts_the_packaged_score_only_when_geometry_is_higher() -> None:
    base = _FlatModel(0.05)
    scorer = PoseGeometryFallScorer(base, frame_width=_WIDTH, frame_height=_HEIGHT)

    lifted = scorer.predict(_window((_UPRIGHT, 12), (_ON_FLOOR, 18)))
    assert lifted.fall_transition == 1.0
    assert lifted.background == 0.0
    assert lifted.fallen == 0.0

    kept = scorer.predict(_window((_CROUCH, 30)))
    assert kept.fall_transition == 0.05

    scorer.warmup()
    assert base.warmed is True
