"""Pose-geometry fall scorer over the pose+bbox56 window.

The packaged proxy classifier does not respond to a real corridor fall (owner
fall 2026-09-15: 83 s continuous track, ``fall_transition`` flat at 0.03-0.05
through the fall). This scorer reads the same ``(30, 56)`` window and names the
transition the model misses: an upright person whose box collapses from tall
to wide while the torso turns horizontal inside the two-second window.

It is composed in front of the packaged model at the runtime model seam and
returns the larger of the two transition scores, so policy votes, episode
promotion and admission are unchanged. ``fallen`` stays 0.0 like the packaged
ONNX runner; the policy owns the fallen lifecycle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Final, Protocol, runtime_checkable

from worker.domains.fall.pose_bbox56 import (
    COCO17_KEYPOINTS,
    POSE_BBOX56_DIM,
)
from worker.interfaces.fall_model import FallV2ModelProtocol, FallV2Probabilities
from worker.types import FallModelInput

POSE_GEOMETRY_SCORER_VERSION: Final = "pose-geometry-v1"

# COCO-17 indices of the torso landmarks the angle is measured between.
_LEFT_SHOULDER: Final = 5
_RIGHT_SHOULDER: Final = 6
_LEFT_HIP: Final = 11
_RIGHT_HIP: Final = 12
_BBOX_OFFSET: Final = COCO17_KEYPOINTS * 3
_VALID_INDEX: Final = POSE_BBOX56_DIM - 1

# Upright evidence is searched over the first half of the window; the collapsed
# state is the median of the last ten rows. A collapse at time C then
# qualifies every window starting in [C-1.33s, C), four classifier ticks, which
# is what the policy's three consecutive votes need. Narrower front edges
# yielded two ticks on the owner's real fall and no alert.
_UPRIGHT_ROWS: Final = 15
_EDGE_ROWS: Final = 10
_MIN_EDGE_ROWS: Final = 5

# Owner fall (corridor, 2026-09-15 04:36Z): aspect 2.7 -> 0.45-0.75 and torso
# 2-6 deg -> 52-68 deg within two seconds. Crouching with a laptop held aspect
# 1.25 / torso 16-20 deg for 50 s; standing up ran the transition backwards.
_UPRIGHT_MIN_ASPECT: Final = 1.5
_UPRIGHT_MAX_TORSO_DEG: Final = 30.0
_ASPECT_DROP_START: Final = 0.3
_ASPECT_DROP_FULL: Final = 0.7
_TORSO_DEG_START: Final = 30.0
_TORSO_DEG_FULL: Final = 60.0
# A caregiver bending over a bed (room 207/210, 2026-09-15 04:51-04:58) also
# collapses the box and turns the torso horizontal, but the legs stay planted:
# ankle-to-hip drop was 0.48-0.73 of the box height while the owner lying on
# the floor measured 0.14-0.19. When the legs are not visible, the box must
# have lost more than half of its standing height instead (0.24 on the fall,
# 0.79 on the bends).
_LEGS_DOWN_MAX_FRAC: Final = 0.3
_HEIGHT_COLLAPSE_MAX_RATIO: Final = 0.45
_LEFT_KNEE: Final = 13
_RIGHT_KNEE: Final = 14
_LEFT_ANKLE: Final = 15
_RIGHT_ANKLE: Final = 16


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _RowGeometry:
    aspect: float
    height: float
    torso_deg: float | None
    leg_frac: float | None


def _unit(value: float, start: float, full: float) -> float:
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


def _row_geometry(row: Sequence[float], frame_aspect_ratio: float) -> _RowGeometry | None:
    if len(row) != POSE_BBOX56_DIM or row[_VALID_INDEX] != 1.0:
        return None
    x1, y1, x2, y2 = row[_BBOX_OFFSET : _BBOX_OFFSET + 4]
    width = (x2 - x1) * frame_aspect_ratio
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    return _RowGeometry(
        aspect=height / width,
        height=height,
        torso_deg=_torso_deg(row, frame_aspect_ratio),
        leg_frac=_leg_frac(row, frame_aspect_ratio, height),
    )


def _landmark(
    row: Sequence[float], index: int, frame_aspect_ratio: float
) -> tuple[float, float] | None:
    x, y, confidence = row[index * 3 : index * 3 + 3]
    # pose_bbox56_row zeroes every triplet under its confidence gate.
    if confidence <= 0.0:
        return None
    return (x * frame_aspect_ratio, y)


def _centre(
    row: Sequence[float], indexes: tuple[int, ...], frame_aspect_ratio: float
) -> tuple[float, float] | None:
    points = [
        point
        for index in indexes
        if (point := _landmark(row, index, frame_aspect_ratio)) is not None
    ]
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _torso_deg(row: Sequence[float], frame_aspect_ratio: float) -> float | None:
    shoulders = _centre(row, (_LEFT_SHOULDER, _RIGHT_SHOULDER), frame_aspect_ratio)
    hips = _centre(row, (_LEFT_HIP, _RIGHT_HIP), frame_aspect_ratio)
    if shoulders is None or hips is None:
        return None
    return math.degrees(math.atan2(abs(hips[0] - shoulders[0]), abs(hips[1] - shoulders[1])))


def _leg_frac(row: Sequence[float], frame_aspect_ratio: float, height: float) -> float | None:
    """How far below the hips the lowest visible leg joint sits, as a share of box height."""
    hips = _centre(row, (_LEFT_HIP, _RIGHT_HIP), frame_aspect_ratio)
    feet = _centre(row, (_LEFT_ANKLE, _RIGHT_ANKLE), frame_aspect_ratio) or _centre(
        row, (_LEFT_KNEE, _RIGHT_KNEE), frame_aspect_ratio
    )
    if hips is None or feet is None:
        return None
    return (feet[1] - hips[1]) / height


def _edge_median(rows: Sequence[_RowGeometry]) -> tuple[float, float | None]:
    aspect = median(row.aspect for row in rows)
    angles = [row.torso_deg for row in rows if row.torso_deg is not None]
    return aspect, (median(angles) if len(angles) >= _MIN_EDGE_ROWS else None)


def _legs_down(late: Sequence[_RowGeometry], standing_height: float) -> bool:
    """The legs lie with the torso, or the whole figure lost most of its height."""
    legs = [row.leg_frac for row in late if row.leg_frac is not None]
    if len(legs) >= _MIN_EDGE_ROWS:
        return median(legs) <= _LEGS_DOWN_MAX_FRAC
    return median(row.height for row in late) / standing_height <= _HEIGHT_COLLAPSE_MAX_RATIO


def _upright_evidence(rows: Sequence[_RowGeometry]) -> tuple[float, float | None]:
    """The most upright the person stood in the span: tallest box, straightest torso."""
    aspect = max(row.aspect for row in rows)
    angles = [row.torso_deg for row in rows if row.torso_deg is not None]
    return aspect, (min(angles) if angles else None)


def geometry_fall_transition(features: FallModelInput, frame_aspect_ratio: float) -> float:
    """Score one window in [0, 1]: upright at the start, collapsed at the end."""
    # The classifier hands the model a (30, 56) window of rows; a flat tuple is
    # not a window and scores nothing.
    rows = [
        geometry
        for row in features
        if isinstance(row, tuple)
        and (geometry := _row_geometry(row, frame_aspect_ratio)) is not None
    ]
    if len(rows) < 2 * _MIN_EDGE_ROWS:
        return 0.0
    early, late = rows[:_UPRIGHT_ROWS], rows[-_EDGE_ROWS:]
    early_aspect, early_torso = _upright_evidence(early)
    late_aspect, late_torso = _edge_median(late)
    if early_aspect < _UPRIGHT_MIN_ASPECT:
        return 0.0
    if early_torso is not None and early_torso > _UPRIGHT_MAX_TORSO_DEG:
        return 0.0
    if late_torso is None:
        return 0.0
    if not _legs_down(late, max(row.height for row in early)):
        return 0.0
    aspect_drop = (early_aspect - late_aspect) / early_aspect
    return _unit(aspect_drop, _ASPECT_DROP_START, _ASPECT_DROP_FULL) * _unit(
        late_torso, _TORSO_DEG_START, _TORSO_DEG_FULL
    )


class PoseGeometryFallScorer:
    """``FallV2ModelProtocol`` that lifts the packaged score by the geometric one."""

    _base: FallV2ModelProtocol
    _frame_aspect_ratio: float

    def __init__(self, base: FallV2ModelProtocol, *, frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        self._base = base
        self._frame_aspect_ratio = frame_width / frame_height

    @property
    def base(self) -> FallV2ModelProtocol:
        return self._base

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        packaged = self._base.predict(features)
        geometric = geometry_fall_transition(features, self._frame_aspect_ratio)
        if geometric <= packaged.fall_transition:
            return packaged
        return FallV2Probabilities(
            background=1.0 - geometric,
            fall_transition=geometric,
            fallen=packaged.fallen,
        )

    def warmup(self) -> None:
        base = self._base
        if isinstance(base, _Warmable):
            base.warmup()


__all__ = [
    "POSE_GEOMETRY_SCORER_VERSION",
    "PoseGeometryFallScorer",
    "geometry_fall_transition",
]
