from __future__ import annotations

import numpy as np

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD
from worker.pipeline.perception.features.window_features import extract_window_features

_FEATURE_DIM = 45
_T_WINDOW = 30


def _all_zero_window() -> np.ndarray:
    return np.zeros((_T_WINDOW, 17, 3), dtype=np.float32)


def _partial_conf_window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    window = rng.random((_T_WINDOW, 17, 3)).astype(np.float32)
    window[:, ::2, 2] = 0.0
    return window


def _above_threshold_window(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    window = rng.random((_T_WINDOW, 17, 3)).astype(np.float32)
    window[:, :, 2] = DEFAULT_FALL_CONFIDENCE_THRESHOLD + 0.1
    return window


class TestExtractWindowFeaturesShape:
    def test_output_shape_is_feature_dim(self) -> None:
        feats = extract_window_features(_all_zero_window())
        assert feats.shape == (_FEATURE_DIM,)

    def test_output_dtype_is_float32(self) -> None:
        feats = extract_window_features(_all_zero_window())
        assert feats.dtype == np.float32

    def test_feature_dim_matches_config_constant(self) -> None:
        feats = extract_window_features(_above_threshold_window())
        assert len(feats) == _FEATURE_DIM


class TestExtractWindowFeaturesNumerical:
    def test_no_nan_inf_on_all_zero_window(self) -> None:
        feats = extract_window_features(_all_zero_window())
        assert np.all(np.isfinite(feats)), (
            f"NaN or inf in features for all-zero window: indices "
            f"{np.where(~np.isfinite(feats))[0].tolist()}"
        )

    def test_no_nan_inf_on_partial_conf_window(self) -> None:
        feats = extract_window_features(_partial_conf_window())
        assert np.all(np.isfinite(feats)), (
            f"NaN or inf in features for partial-conf window: indices "
            f"{np.where(~np.isfinite(feats))[0].tolist()}"
        )

    def test_all_zero_window_produces_all_zero_features(self) -> None:
        feats = extract_window_features(_all_zero_window())
        np.testing.assert_array_equal(feats, 0.0)

    def test_active_window_features_are_non_negative(self) -> None:
        feats = extract_window_features(_above_threshold_window())
        assert np.all(feats >= 0.0), (
            f"Negative feature values at indices: {np.where(feats < 0)[0].tolist()}"
        )
