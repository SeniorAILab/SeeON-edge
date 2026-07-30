"""Window-level feature extraction for fall detection.

Single source of truth for FEATURE_DIM.  The constant ``FEATURE_DIM`` in
``training.config`` is set to ``_D`` defined here; do not derive D from
anywhere else.

Feature breakdown  (D = 45)
---------------------------
00-33  Per-keypoint velocity: mean |Δx|, mean |Δy| per keypoint   (17 × 2 = 34)
34-35  Centroid displacement: total, max single-step                        (2)
36-38  Bounding-box aspect-ratio per frame: mean, std, range               (3)
39     Torso tilt |sin θ| vs vertical: mean over valid frames              (1)
40-41  Centroid vertical drop velocity: mean |Δy|, max |Δy|               (2)
42-43  Torso-vertical |shoulder_mid_y − hip_mid_y|: mean, std             (2)
44     Overall motion energy: Σ squared centroid-step distances            (1)

Keypoint coordinates are assumed to be normalised [0, 1] values (x/w, y/h)
exactly as stored in the npz cache by ``extract_poses``.  All features are
NaN-free; any feature whose computation yields no valid frames emits 0.0.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

# COCO-17 indices — mirrors demo/features.py constants exactly.
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12

_D: int = 45  # total feature dimension; config.FEATURE_DIM is set to this value


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract hand-crafted fall-detection features from a pose window.

    Parameters
    ----------
    window:
        float32[T, 17, 3] — T frames of COCO-17 keypoints.
        The last dimension is (x, y, conf).
        Keypoints with conf == 0 are treated as MISSING and skipped.

    Returns
    -------
    np.ndarray
        float32[D=45] feature vector.  Never contains NaN or inf.
    """
    window = np.asarray(window, dtype=np.float32)
    T, N_KPT, _ = window.shape

    x = window[:, :, 0]  # [T, 17]
    y = window[:, :, 1]  # [T, 17]
    conf = window[:, :, 2]  # [T, 17]
    valid = conf > DEFAULT_FALL_CONFIDENCE_THRESHOLD  # [T, 17] bool mask

    features: list[float] = []

    # ------------------------------------------------------------------
    # 1. Per-keypoint velocity (34 features: 17 kpts × [mean|Δx|, mean|Δy|])
    # A step is valid only when both adjacent frames have conf > threshold.
    # ------------------------------------------------------------------
    valid_step = valid[:-1] & valid[1:]  # [T-1, 17]
    abs_dx = np.abs(np.diff(x, axis=0))  # [T-1, 17]
    abs_dy = np.abs(np.diff(y, axis=0))  # [T-1, 17]
    for k in range(N_KPT):
        mask = valid_step[:, k]
        if mask.any():
            features.append(float(abs_dx[:, k][mask].mean()))
            features.append(float(abs_dy[:, k][mask].mean()))
        else:
            features.extend([0.0, 0.0])

    # ------------------------------------------------------------------
    # 2. Centroid displacement (2 features: total, max single-step)
    # Centroid is the mean position of all valid keypoints per frame.
    # Frames with zero valid keypoints fall back to centroid = (0, 0).
    # ------------------------------------------------------------------
    valid_counts = valid.sum(axis=1).astype(np.float32)  # [T]
    denom = np.maximum(valid_counts, 1.0)  # prevent div-by-zero
    cx = np.where(valid_counts > 0, (x * valid).sum(axis=1) / denom, 0.0)
    cy = np.where(valid_counts > 0, (y * valid).sum(axis=1) / denom, 0.0)

    step_dists = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)  # [T-1]
    features.append(float(step_dists.sum()))
    features.append(float(step_dists.max()) if len(step_dists) > 0 else 0.0)

    # ------------------------------------------------------------------
    # 3. Bounding-box aspect-ratio (3 features: mean, std, range)
    # bbox = axis-aligned bounding box of all valid keypoints per frame.
    # ------------------------------------------------------------------
    aspect_ratios: list[float] = []
    for t in range(T):
        mask_t = valid[t]
        if int(mask_t.sum()) >= 2:
            w = float(x[t, mask_t].max() - x[t, mask_t].min())
            h = float(y[t, mask_t].max() - y[t, mask_t].min())
            aspect_ratios.append(w / h if h > 0.0 else 0.0)

    if aspect_ratios:
        ar = np.array(aspect_ratios, dtype=np.float32)
        features.extend([float(ar.mean()), float(ar.std()), float(ar.max() - ar.min())])
    else:
        features.extend([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # 4. Torso tilt |sin θ| vs vertical (1 feature: mean over valid frames)
    # θ = angle of vector (shoulder_mid → hip_mid) from the vertical axis.
    # |sin θ| = |horizontal component| / |vector|.
    # Small value → person upright; value near 1 → person horizontal (fallen).
    # Mirrors torso_vertical semantics from demo/features.py.
    # ------------------------------------------------------------------
    tilt_vals: list[float] = []
    for t in range(T):
        sx_vals = [float(x[t, k]) for k in (_LEFT_SHOULDER, _RIGHT_SHOULDER) if valid[t, k]]
        sy_vals = [float(y[t, k]) for k in (_LEFT_SHOULDER, _RIGHT_SHOULDER) if valid[t, k]]
        hx_vals = [float(x[t, k]) for k in (_LEFT_HIP, _RIGHT_HIP) if valid[t, k]]
        hy_vals = [float(y[t, k]) for k in (_LEFT_HIP, _RIGHT_HIP) if valid[t, k]]
        if sx_vals and hx_vals:
            sx_mid = sum(sx_vals) / len(sx_vals)
            sy_mid = sum(sy_vals) / len(sy_vals)
            hx_mid = sum(hx_vals) / len(hx_vals)
            hy_mid = sum(hy_vals) / len(hy_vals)
            dx = hx_mid - sx_mid
            dy = hy_mid - sy_mid
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 0.0:
                tilt_vals.append(abs(dx) / mag)

    features.append(float(np.mean(tilt_vals)) if tilt_vals else 0.0)

    # ------------------------------------------------------------------
    # 5. Centroid vertical drop velocity (2 features: mean |Δy|, max |Δy|)
    # Uses the same centroid cy computed above.
    # ------------------------------------------------------------------
    abs_dcy = np.abs(np.diff(cy))  # [T-1]
    features.append(float(abs_dcy.mean()) if len(abs_dcy) > 0 else 0.0)
    features.append(float(abs_dcy.max()) if len(abs_dcy) > 0 else 0.0)

    # ------------------------------------------------------------------
    # 6. Torso-vertical |shoulder_mid_y − hip_mid_y| (2 features: mean, std)
    # Mirrors demo/features.py torso_vertical (without frame-height division;
    # operates in raw pixel coordinates from the npz cache).
    # ------------------------------------------------------------------
    tv_vals: list[float] = []
    for t in range(T):
        s_ys = [float(y[t, k]) for k in (_LEFT_SHOULDER, _RIGHT_SHOULDER) if valid[t, k]]
        h_ys = [float(y[t, k]) for k in (_LEFT_HIP, _RIGHT_HIP) if valid[t, k]]
        if s_ys and h_ys:
            tv_vals.append(abs(sum(s_ys) / len(s_ys) - sum(h_ys) / len(h_ys)))

    if tv_vals:
        tv = np.array(tv_vals, dtype=np.float32)
        features.extend([float(tv.mean()), float(tv.std())])
    else:
        features.extend([0.0, 0.0])

    # ------------------------------------------------------------------
    # 7. Overall motion energy (1 feature)
    # Sum of squared centroid-step distances — captures gross movement.
    # ------------------------------------------------------------------
    features.append(float((step_dists**2).sum()))

    result = np.array(features, dtype=np.float32)
    assert len(result) == _D, f"Feature dimension mismatch: expected {_D}, got {len(result)}"
    return result
