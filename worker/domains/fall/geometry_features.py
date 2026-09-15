"""Window features and physical gate for the trained fall scorer.

The deployed proxy classifier scored a real corridor fall at 0.05 (owner fall,
2026-09-15), and on the sealed AI-Hub split it promoted no fall at all. A model
trained on that same dataset over these fourteen window features detects it, so
the features are a published contract: the training set and this module must
compute them identically or the served scores mean nothing.

``collapse_gate`` keeps the physical precondition the dataset could not teach.
AI-Hub has no caregiver bending over a bed, so a model trained on it alone
called that a fall; the gate refuses any window whose legs stayed planted or
whose hips sat at the bottom of the box.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median, pstdev
from typing import Final

from worker.domains.fall.geometry_scorer import (
    EDGE_ROWS,
    MIN_WINDOW_ROWS,
    UPRIGHT_ROWS,
    RowGeometry,
    hips_mid_box,
    legs_down,
    row_geometry,
)
from worker.types import FallModelInput

GEOMETRY_FEATURE_DIM: Final = 14
# The gate reads two edges of the window; below this many valid rows the window
# describes too little of the person to assert anything.
_MIN_GATE_ROWS: Final = 20


def _median_or_zero(values: Sequence[float | None]) -> float:
    present = [value for value in values if value is not None]
    return median(present) if present else 0.0


def window_geometry(window: FallModelInput, frame_aspect_ratio: float) -> list[RowGeometry]:
    """Valid per-row geometry for one classifier window, in time order."""
    return [
        geometry
        for row in window
        if isinstance(row, tuple)
        and (geometry := row_geometry(row, frame_aspect_ratio)) is not None
    ]


def geometry_features(window: FallModelInput, frame_aspect_ratio: float) -> list[float]:
    """The published fourteen-feature vector for one window.

    Order is part of the trained model's contract. A window with too few valid
    rows yields zeros rather than a guess.
    """
    rows = window_geometry(window, frame_aspect_ratio)
    if len(rows) < MIN_WINDOW_ROWS:
        return [0.0] * GEOMETRY_FEATURE_DIM
    early, late = rows[:UPRIGHT_ROWS], rows[-EDGE_ROWS:]

    early_aspect = max(row.aspect for row in early)
    late_aspect = _median_or_zero([row.aspect for row in late])
    early_height = max(row.height for row in early)
    late_height = _median_or_zero([row.height for row in late])
    early_torso = _median_or_zero([row.torso_deg for row in early])
    late_torso = _median_or_zero([row.torso_deg for row in late])
    early_leg = _median_or_zero([row.leg_frac for row in early])
    late_leg = _median_or_zero([row.leg_frac for row in late])
    late_hip = _median_or_zero([row.hip_frac for row in late])
    aspects = [row.aspect for row in rows]
    return [
        early_aspect,
        late_aspect,
        (early_aspect - late_aspect) / early_aspect if early_aspect else 0.0,
        early_height,
        late_height,
        1.0 - (late_height / early_height if early_height else 1.0),
        early_torso,
        late_torso,
        late_torso - early_torso,
        early_leg,
        late_leg,
        early_leg - late_leg,
        late_hip,
        pstdev(aspects) if len(aspects) > 1 else 0.0,
    ]


def collapse_gate(window: FallModelInput, frame_aspect_ratio: float) -> bool:
    """Whether the window is physically a person going down rather than bending."""
    rows = window_geometry(window, frame_aspect_ratio)
    if len(rows) < _MIN_GATE_ROWS:
        return False
    early, late = rows[:UPRIGHT_ROWS], rows[-EDGE_ROWS:]
    standing = max(row.height for row in early)
    return legs_down(late, standing) and hips_mid_box(late)


__all__ = [
    "GEOMETRY_FEATURE_DIM",
    "collapse_gate",
    "geometry_features",
    "window_geometry",
]
