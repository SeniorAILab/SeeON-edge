"""Fall transition scored by the trained geometry classifier.

Evidence for replacing the packaged proxy at this seam (2026-09-15): on the
sealed AI-Hub split the proxy promoted 0 of 60 falls (average precision 0.050
against a 0.058 positive rate), and it scored the owner's real corridor fall at
0.05 throughout. The classifier trained on the same dataset plus our own
recorded negatives promotes both real corridor falls.

The physical gate stays in front of the model: AI-Hub has no caregiver bending
over a bed, and the ungated model called that a fall eleven times in twenty
minutes of room 207.
"""

from __future__ import annotations

from typing import Final

from worker.domains.fall.geometry_features import (
    GEOMETRY_FEATURE_DIM,
    collapse_gate,
    geometry_features,
)
from worker.interfaces.fall_model import FallV2Probabilities
from worker.interfaces.vector_classifier import VectorClassifierProtocol
from worker.types import FallModelInput

TRAINED_FALL_SCORER_VERSION: Final = "geometry-rf-v1"


class TrainedGeometryFallScorer:
    """``FallV2ModelProtocol`` backed by the trained geometry classifier."""

    _classifier: VectorClassifierProtocol
    _frame_aspect_ratio: float

    def __init__(
        self,
        classifier: VectorClassifierProtocol,
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if classifier.feature_dim != GEOMETRY_FEATURE_DIM:
            raise ValueError(
                f"classifier takes {classifier.feature_dim} features, "
                f"the window publishes {GEOMETRY_FEATURE_DIM}"
            )
        self._classifier = classifier
        self._frame_aspect_ratio = frame_width / frame_height

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        if not collapse_gate(features, self._frame_aspect_ratio):
            return FallV2Probabilities(1.0, 0.0, 0.0)
        vector = geometry_features(features, self._frame_aspect_ratio)
        transition = self._classifier.positive_probability(vector)
        # The policy owns the fallen lifecycle; this model scores the transition.
        return FallV2Probabilities(1.0 - transition, transition, 0.0)

    def warmup(self) -> None:
        _ = self.predict(tuple((0.0,) * 56 for _ in range(30)))


__all__ = ["TRAINED_FALL_SCORER_VERSION", "TrainedGeometryFallScorer"]
