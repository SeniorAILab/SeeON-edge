"""Fall transition scored by the trained geometry classifier.

Evidence for replacing the rule-based geometry scorer (#542/#575) as the
fall-alert authority (2026-09-24): on site replay over all recorded camera
traces, the geometry rule missed both of the owner's real corridor falls at
every threshold tried. A tree ensemble over the same fourteen window-geometry
features (``geometry_features``), trained on the AI-Hub proxy dataset plus
our own recorded site negatives (owner-fall windows excluded), promotes both.
See ``models/fall/geometry-trained-v1/receipt.json`` for the exact training
run this scorer is verified against, and ``scripts/train_fall_geometry_classifier.py``
for how to reproduce it.

This scorer is the fall-alert authority: its ``fall_transition`` is the only
score the fall policy acts on. Both other scorers are comparison-only and
never gate or emit:

* ``shadow_fall_transition`` -- the packaged pose+bbox56 proxy (P542).
* ``shadow_geometry_fall_transition`` -- the rule-based geometry transition
  it replaces (P542/#575).

The physical collapse gate (``geometry_features.collapse_gate``) stays in
front of the model: AI-Hub has no caregiver bending over a bed, and the
ungated model called that a fall repeatedly on site footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_vector_classifier import OrtVectorClassifier
from worker.adapters.model.pose_bbox56_bundle_support import read_json
from worker.domains.fall.geometry_features import (
    GEOMETRY_FEATURE_DIM,
    collapse_gate,
    geometry_features,
)
from worker.domains.fall.geometry_scorer import geometry_fall_transition
from worker.interfaces.fall_model import FallModelProtocol, FallProbabilities
from worker.interfaces.vector_classifier import VectorClassifierProtocol
from worker.types import FallModelInput

TRAINED_FALL_SCORER_VERSION: Final = "geometry-trained-v1"

_MODEL_FILENAME: Final = "model.onnx"
_RECEIPT_FILENAME: Final = "receipt.json"


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingReceipt:
    """The identity fields the registry's threshold precedence reads (ADR "threshold
    precedence: receipt threshold is authoritative").

    The full receipt on disk carries far more (dataset revision, site-trace
    sha256s, hyperparameters, training commit, library versions) for
    reproducibility and audit; only these fields are load-bearing for serving.
    """

    model_sha256: str
    threshold: float
    promotion_eligible: bool
    feature_version: str

    @classmethod
    def from_dict(cls, document: object) -> TrainingReceipt:
        if not isinstance(document, dict):
            raise ModelLoadError("invalid receipt.json")
        model_sha256 = document.get("model_sha256")
        threshold = document.get("threshold")
        promotion_eligible = document.get("promotion_eligible")
        feature_version = document.get("feature_version")
        if (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or not isinstance(threshold, int | float)
            or isinstance(threshold, bool)
            or not isinstance(promotion_eligible, bool)
            or not isinstance(feature_version, str)
            or not feature_version
        ):
            raise ModelLoadError("receipt.json is missing a required field")
        return cls(model_sha256, float(threshold), promotion_eligible, feature_version)


class TrainedGeometryFallScorer:
    """``FallModelProtocol`` backed by the trained geometry classifier.

    Implements the registry's structural ``_ArtifactProvenance``/``_ThresholdReceipt``
    protocols (``worker/domains/registry.py``) so ``_audit_snapshot`` reports this
    model's own artifact digest and receipt threshold, not the packaged proxy's.
    """

    _base: FallModelProtocol
    _classifier: VectorClassifierProtocol
    _frame_aspect_ratio: float
    _gated: bool
    artifact_digest: str
    receipt_threshold: float | None
    promotion_eligible: bool

    def __init__(
        self,
        base: FallModelProtocol,
        classifier: VectorClassifierProtocol,
        *,
        frame_width: int,
        frame_height: int,
        receipt: TrainingReceipt,
        gated: bool = True,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if classifier.feature_dim != GEOMETRY_FEATURE_DIM:
            raise ModelLoadError(
                f"classifier takes {classifier.feature_dim} features, "
                f"the window publishes {GEOMETRY_FEATURE_DIM}"
            )
        artifact_digest = getattr(classifier, "artifact_digest", None)
        if artifact_digest != receipt.model_sha256:
            # Belt and suspenders: OrtVectorClassifier.from_model_path already
            # verifies this when given expected_digest; a mismatch here means
            # the classifier and receipt were paired from different sources.
            raise ModelLoadError("classifier artifact digest does not match receipt.json")
        self._base = base
        self._classifier = classifier
        self._frame_aspect_ratio = frame_width / frame_height
        self._gated = gated
        self.artifact_digest = artifact_digest
        self.receipt_threshold = receipt.threshold
        self.promotion_eligible = receipt.promotion_eligible

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        base: FallModelProtocol,
        *,
        frame_width: int,
        frame_height: int,
        gated: bool = True,
    ) -> TrainedGeometryFallScorer:
        directory = artifact_dir.expanduser()
        receipt = TrainingReceipt.from_dict(read_json(directory / _RECEIPT_FILENAME))
        classifier = OrtVectorClassifier.from_model_path(
            directory / _MODEL_FILENAME,
            feature_dim=GEOMETRY_FEATURE_DIM,
            expected_digest=receipt.model_sha256,
        )
        return cls(
            base,
            classifier,
            frame_width=frame_width,
            frame_height=frame_height,
            receipt=receipt,
            gated=gated,
        )

    def predict(self, features: FallModelInput) -> FallProbabilities:
        packaged = self._base.predict(features)
        geometric = geometry_fall_transition(features, self._frame_aspect_ratio)
        gate_passed = collapse_gate(features, self._frame_aspect_ratio)
        if self._gated and not gate_passed:
            transition = 0.0
        else:
            vector = geometry_features(features, self._frame_aspect_ratio)
            transition = self._classifier.positive_probability(vector)
        return FallProbabilities(
            background=1.0 - transition,
            fall_transition=transition,
            fallen=packaged.fallen,
            model_evidence=packaged.model_evidence,
            shadow_fall_transition=packaged.fall_transition,
            shadow_geometry_fall_transition=geometric,
        )

    def warmup(self) -> None:
        base = self._base
        if isinstance(base, _Warmable):
            base.warmup()
        _ = self.predict(tuple((0.0,) * 56 for _ in range(30)))


__all__ = ["TRAINED_FALL_SCORER_VERSION", "TrainedGeometryFallScorer", "TrainingReceipt"]
