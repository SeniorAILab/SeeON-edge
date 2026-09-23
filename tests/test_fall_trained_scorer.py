"""Trained geometry classifier: the serving seam under ``TrainedGeometryFallScorer``.

Feature parity with training, fail-fast artifact verification (ADR-0002), and
the shadow-only status of the other two scores (never gate, never emit) are
exactly the properties the fall-alert authority migration depends on -- see
``worker/domains/fall/trained_scorer.py`` and ``worker/domains/fall/geometry_features.py``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_vector_classifier import OrtVectorClassifier
from worker.domains.fall.classifier import FALL_WINDOW_FRAMES
from worker.domains.fall.geometry_features import GEOMETRY_FEATURE_DIM
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.domains.fall.trained_scorer import TrainedGeometryFallScorer, TrainingReceipt
from worker.interfaces.fall_model import FallProbabilities
from worker.types import FallModelInput

_WIDTH, _HEIGHT = 640, 360


def _row(
    bbox: tuple[float, float, float, float], torso_deg: float, *, legs_down: bool = True
) -> tuple[float, ...]:
    """A person box with shoulders and hips at ``torso_deg`` from vertical.

    Mirrors ``tests/test_fall_geometry_scorer.py``'s helper of the same name --
    both build the same fourteen-feature geometry from the same pose+bbox56
    row shape, so the fixtures are worth keeping identical in behavior.
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


_UPRIGHT = _row((300.0, 100.0, 360.0, 300.0), 5.0, legs_down=False)
_ON_FLOOR = _row((250.0, 240.0, 410.0, 300.0), 65.0)
# Caregiver bending over a bed: box collapses and torso turns horizontal, but
# the legs stay planted below the hips -- the exact shape collapse_gate exists
# to reject (see geometry_features.py's module docstring).
_BENDING = _row((270.0, 190.0, 400.0, 300.0), 70.0, legs_down=False)


def _window(*segments: tuple[tuple[float, ...], int]) -> FallModelInput:
    rows: list[tuple[float, ...]] = []
    for row, count in segments:
        rows.extend([row] * count)
    assert len(rows) == FALL_WINDOW_FRAMES
    return tuple(rows)


# --------------------------------------------------------------------------
# Training/serving feature parity
# --------------------------------------------------------------------------


def test_training_and_serving_call_the_identical_feature_function() -> None:
    """scripts/train_fall_geometry_classifier.py imports ``geometry_features``
    directly from this module rather than reimplementing it -- there is no
    second copy of the fourteen-feature contract to drift out of sync."""
    from scripts.train_fall_geometry_classifier import geometry_features as training_feature_fn
    from worker.domains.fall.geometry_features import geometry_features as serving_feature_fn

    assert training_feature_fn is serving_feature_fn


# --------------------------------------------------------------------------
# TrainingReceipt parsing
# --------------------------------------------------------------------------

_VALID_RECEIPT_DOC = {
    "model_sha256": "a" * 64,
    "threshold": 0.2,
    "promotion_eligible": True,
    "feature_version": "geometry-features-v1",
}


def test_receipt_parses_a_valid_document() -> None:
    receipt = TrainingReceipt.from_dict(_VALID_RECEIPT_DOC)
    assert receipt.model_sha256 == "a" * 64
    assert receipt.threshold == 0.2
    assert receipt.promotion_eligible is True
    assert receipt.feature_version == "geometry-features-v1"


@pytest.mark.parametrize(
    "field", ["model_sha256", "threshold", "promotion_eligible", "feature_version"]
)
def test_receipt_rejects_a_document_missing_a_required_field(field: str) -> None:
    document = dict(_VALID_RECEIPT_DOC)
    del document[field]
    with pytest.raises(ModelLoadError):
        TrainingReceipt.from_dict(document)


def test_receipt_rejects_a_non_dict_document() -> None:
    with pytest.raises(ModelLoadError):
        TrainingReceipt.from_dict("not a dict")


# --------------------------------------------------------------------------
# OrtVectorClassifier: configured-but-unloadable refuses boot (ADR-0002)
# --------------------------------------------------------------------------


class _FakeInput:
    def __init__(self, name: str, shape: list[int | None]) -> None:
        self.name = name
        self.shape = shape


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Stands in for onnxruntime's InferenceSession -- exercises the digest,
    shape and output-name checks without an ONNX runtime or a real model."""

    def __init__(self, feature_dim: int, *, positive_probability: float = 0.7) -> None:
        self._feature_dim = feature_dim
        self._positive_probability = positive_probability

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput("X", [None, self._feature_dim])]

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput("probabilities")]

    def run(self, output_names: list[str] | None, input_feed: dict[str, object]) -> list[object]:
        del output_names, input_feed
        return [np.array([[1.0 - self._positive_probability, self._positive_probability]])]


def _fake_session_factory(feature_dim: int, *, positive_probability: float = 0.7):
    def factory(model_path: str, providers: list[str]) -> _FakeSession:
        del model_path, providers
        return _FakeSession(feature_dim, positive_probability=positive_probability)

    return factory


def test_digest_mismatch_refuses_to_load(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stand-in bytes, never parsed as ONNX by this test")
    with pytest.raises(ModelLoadError, match="digest mismatch"):
        OrtVectorClassifier.from_model_path(
            model_path,
            feature_dim=GEOMETRY_FEATURE_DIM,
            expected_digest="0" * 64,
            session_factory=_fake_session_factory(GEOMETRY_FEATURE_DIM),
        )


def test_matching_digest_loads_and_scores(tmp_path: Path) -> None:
    payload = b"stand-in bytes, never parsed as ONNX by this test"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    classifier = OrtVectorClassifier.from_model_path(
        model_path,
        feature_dim=GEOMETRY_FEATURE_DIM,
        expected_digest=digest,
        session_factory=_fake_session_factory(GEOMETRY_FEATURE_DIM, positive_probability=0.42),
    )
    assert classifier.artifact_digest == digest
    assert classifier.positive_probability([0.0] * GEOMETRY_FEATURE_DIM) == pytest.approx(0.42)


# --------------------------------------------------------------------------
# TrainedGeometryFallScorer: authority, shadows, and the collapse-gate toggle
# --------------------------------------------------------------------------


class _FakeClassifier:
    feature_dim: int = GEOMETRY_FEATURE_DIM

    def __init__(self, score: float, *, digest: str = "d" * 64) -> None:
        self._score = score
        self.artifact_digest = digest

    def positive_probability(self, vector: object) -> float:
        assert len(vector) == GEOMETRY_FEATURE_DIM  # type: ignore[arg-type]
        return self._score


class _FlatBase:
    def __init__(self, transition: float) -> None:
        self.transition = transition
        self.warmed = False

    def predict(self, features: FallModelInput) -> FallProbabilities:
        del features
        return FallProbabilities(1.0 - self.transition, self.transition, 0.0)

    def warmup(self) -> None:
        self.warmed = True


def _receipt(
    *, digest: str = "d" * 64, threshold: float = 0.2, promotion_eligible: bool = True
) -> TrainingReceipt:
    return TrainingReceipt(
        model_sha256=digest,
        threshold=threshold,
        promotion_eligible=promotion_eligible,
        feature_version="geometry-features-v1",
    )


def test_mismatched_classifier_and_receipt_digests_refuse_to_load() -> None:
    classifier = _FakeClassifier(0.9, digest="a" * 64)
    with pytest.raises(ModelLoadError, match="does not match receipt"):
        TrainedGeometryFallScorer(
            _FlatBase(0.0),
            classifier,
            frame_width=_WIDTH,
            frame_height=_HEIGHT,
            receipt=_receipt(digest="b" * 64),
        )


def test_rejects_non_positive_frame_dimensions() -> None:
    with pytest.raises(ValueError, match="frame dimensions"):
        TrainedGeometryFallScorer(
            _FlatBase(0.0),
            _FakeClassifier(0.0),
            frame_width=0,
            frame_height=_HEIGHT,
            receipt=_receipt(),
        )


def test_rejects_a_classifier_whose_feature_dim_does_not_match_the_window() -> None:
    classifier = _FakeClassifier(0.0)
    classifier.feature_dim = GEOMETRY_FEATURE_DIM + 1
    with pytest.raises(ModelLoadError, match="features"):
        TrainedGeometryFallScorer(
            _FlatBase(0.0),
            classifier,
            frame_width=_WIDTH,
            frame_height=_HEIGHT,
            receipt=_receipt(),
        )


def test_packaged_score_never_gates_only_the_classifier_score_reaches_the_policy() -> None:
    """The packaged pose+bbox56 proxy (P542) is recorded as ``shadow_fall_transition``
    only; a high packaged score must never leak into the emitted ``fall_transition``."""
    base = _FlatBase(0.99)
    classifier = _FakeClassifier(0.05)
    scorer = TrainedGeometryFallScorer(
        base, classifier, frame_width=_WIDTH, frame_height=_HEIGHT, receipt=_receipt()
    )
    window = _window((_UPRIGHT, 12), (_ON_FLOOR, 18))  # passes the collapse gate

    result = scorer.predict(window)

    assert result.fall_transition == pytest.approx(0.05)
    assert result.background == pytest.approx(0.95)
    assert result.shadow_fall_transition == pytest.approx(0.99)


def test_shadow_geometry_score_is_recorded_but_the_classifier_is_the_authority() -> None:
    """The rule-based geometry transition it replaces (#575) stays visible as
    ``shadow_geometry_fall_transition`` for comparison, but never composes in."""
    classifier = _FakeClassifier(0.0)  # the trained classifier disagrees with the rule
    scorer = TrainedGeometryFallScorer(
        _FlatBase(0.0), classifier, frame_width=_WIDTH, frame_height=_HEIGHT, receipt=_receipt()
    )
    window = _window((_UPRIGHT, 12), (_ON_FLOOR, 18))  # the geometry rule scores this 1.0

    result = scorer.predict(window)

    assert result.shadow_geometry_fall_transition == 1.0
    assert result.fall_transition == 0.0


def test_gated_variant_zeroes_a_high_classifier_score_without_a_physical_collapse() -> None:
    classifier = _FakeClassifier(0.95)
    scorer = TrainedGeometryFallScorer(
        _FlatBase(0.0),
        classifier,
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
        receipt=_receipt(),
        gated=True,
    )
    bending = _window((_BENDING, 30))  # legs stay planted -- fails collapse_gate

    assert scorer.predict(bending).fall_transition == 0.0


def test_ungated_variant_lets_the_classifier_score_through_without_the_gate() -> None:
    classifier = _FakeClassifier(0.95)
    scorer = TrainedGeometryFallScorer(
        _FlatBase(0.0),
        classifier,
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
        receipt=_receipt(),
        gated=False,
    )
    bending = _window((_BENDING, 30))

    assert scorer.predict(bending).fall_transition == pytest.approx(0.95)


def test_artifact_digest_and_receipt_fields_are_exposed_for_the_registry() -> None:
    """worker/domains/registry.py's ``_ArtifactProvenance``/``_ThresholdReceipt``
    protocols read these structurally -- no registry.py change is needed for
    ``_audit_snapshot`` to report this model's own digest and threshold."""
    scorer = TrainedGeometryFallScorer(
        _FlatBase(0.0),
        _FakeClassifier(0.5),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
        receipt=_receipt(digest="d" * 64, threshold=0.2, promotion_eligible=True),
    )

    assert scorer.artifact_digest == "d" * 64
    assert scorer.receipt_threshold == 0.2
    assert scorer.promotion_eligible is True


def test_warmup_delegates_to_the_base_and_scores_a_flat_window() -> None:
    base = _FlatBase(0.1)
    scorer = TrainedGeometryFallScorer(
        base, _FakeClassifier(0.0), frame_width=_WIDTH, frame_height=_HEIGHT, receipt=_receipt()
    )

    scorer.warmup()

    assert base.warmed is True
