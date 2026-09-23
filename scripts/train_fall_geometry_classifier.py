"""Reproducible training entrypoint for the trained fall-geometry classifier.

Produces ``models/fall/geometry-trained-v1/{model.onnx,receipt.json}`` (both
gitignored -- see ``worker/runtime/config/local_env.py``'s
``ML_WORKER_FALL_GEOMETRY_CLASSIFIER_DIR`` for how a worker boot picks them up).

Run from the repo root, with the project's own venv active plus two ephemeral
extras this training-only script needs (the served worker never imports
either):

    PYTHONPATH=. uv run --with pyarrow --with skl2onnx python3 \
        scripts/train_fall_geometry_classifier.py

Feature parity with serving is structural, not asserted: this script imports
``worker.domains.fall.geometry_features.geometry_features`` directly, the
exact function ``TrainedGeometryFallScorer`` calls at inference time. There is
no second implementation to drift out of sync with the first.

Positive/negative examples come from two sources:

* AI-Hub proxy dataset (``Berom0227/seeon-dataset-v0.1.0-source-proxy``,
  pinned revision below): the ``train`` split only. ``sealed_test`` is never
  touched here -- it is read only by the separate replay-validation script's
  reported sealed-eval numbers, computed from this training run's own
  ``receipt.json`` artifact, not by this script re-touching training data.
* Site trace negatives: the first 75% (by wall-clock duration) of each of the
  13 ``/tmp/traces-all/*.jsonl`` recordings, harvested by replaying each trace
  through the real production pipeline (``worker.replay.engine.replay``) with
  a recording stand-in model, so every harvested window is exactly the
  ``(30, 56)`` window ``FallWindowClassifier`` would have built in the field
  -- no separate resampling/windowing reimplementation to get wrong. The
  camera with the two real owner falls (``d9c8ada04907a3e8.jsonl``) has both
  ``t0``-relative ±20s windows (398s, 1695s) excised before replay, as two
  separate replay calls either side of the excised span, so no training
  window ever straddles or touches an owner-fall window. The last 25% of
  every trace is held out entirely (unseen negatives for the replay-based
  validation script, not used here).

The written ``receipt.json``'s ``threshold`` is a placeholder (0.5) at the
end of this script; the replay-validation script is what picks the real
operating threshold from site behavior, and that threshold gets written back
into this same ``receipt.json`` afterward without retraining (the ONNX bytes,
and their sha256, do not change).
"""

from __future__ import annotations

import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import sklearn
from sklearn.ensemble import RandomForestClassifier

from contracts.replay_trace import decode_jsonl
from shared.detection_policies import FallPolicyV2, make_effective_policy
from worker.domains.fall.geometry_features import GEOMETRY_FEATURE_DIM, geometry_features
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.interfaces.fall_model import FallProbabilities
from worker.replay.engine import replay

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "models" / "fall" / "geometry-trained-v1"

SEED = 0
ASPECT = 16 / 9
WINDOW, STRIDE, GUARD = 30, 5, 15
FEATURE_VERSION = "geometry-features-v1"
PLACEHOLDER_THRESHOLD = 0.5
UNAPPLIED_POLICY_THRESHOLD = 0.2

HF_REPO = "Berom0227/seeon-dataset-v0.1.0-source-proxy"
HF_REVISION = "765557189d65c76ce41c22b5aa07a2718fa86077"

TRACES_DIR = Path("/tmp/traces-all")
OWNER_FALL_FILE = "d9c8ada04907a3e8.jsonl"
# (center_s, half_width_s), both t0-relative -- t0 is that file's first row's pts_ns.
OWNER_FALL_WINDOWS_S = ((398.0, 20.0), (1695.0, 20.0))
HELD_OUT_FRACTION = 0.25

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# AI-Hub proxy dataset
# --------------------------------------------------------------------------


def _clip_parquet_path() -> Path:
    pattern = str(
        Path.home()
        / ".cache/huggingface/hub/datasets--Berom0227--seeon-dataset-v0.1.0-source-proxy"
        / "snapshots"
        / HF_REVISION
        / "clips.parquet"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"clips.parquet not found for pinned revision {HF_REVISION}")
    return Path(matches[0])


def _clip_windows(clip: dict) -> list[tuple[tuple[tuple[float, ...], ...], int]]:
    """Same convention as the reference ``fall_eval.py`` this replaces."""
    width, height = int(clip["width"]), int(clip["height"])
    rows = []
    for pose, box in zip(clip["pose"], clip["pose_head_bbox"], strict=False):
        keypoints = [(float(x) * width, float(y) * height, float(c)) for x, y, c in pose]
        bbox = None
        if box is not None and len(box) >= 4:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
        rows.append(pose_bbox56_row(keypoints, bbox, width, height))

    label = clip["labels"]["source_proxy_label"]
    interval = clip["labels"].get("source_proxy_interval_15fps") or {}
    start, end = int(interval.get("start", -1)), int(interval.get("end", -1))

    out: list[tuple[tuple[tuple[float, ...], ...], int]] = []
    for i in range(0, len(rows) - WINDOW + 1, STRIDE):
        last = i + WINDOW - 1
        if label == "positive" and start >= 0:
            if start <= last <= end:
                out.append((tuple(rows[i : i + WINDOW]), 1))
            elif not (start - GUARD <= last <= end + GUARD):
                out.append((tuple(rows[i : i + WINDOW]), 0))
        else:
            out.append((tuple(rows[i : i + WINDOW]), 0))
    return out


def _load_aihub_split(by_split: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    clips = by_split.get(split, [])
    features, labels = [], []
    for clip in clips:
        for window, label in _clip_windows(clip):
            features.append(geometry_features(window, ASPECT))
            labels.append(label)
    x = (
        np.array(features, dtype=np.float32)
        if features
        else np.zeros((0, GEOMETRY_FEATURE_DIM), dtype=np.float32)
    )
    y = np.array(labels, dtype=np.int64)
    return x, y


# --------------------------------------------------------------------------
# Site trace negatives, harvested through the real replay pipeline
# --------------------------------------------------------------------------


class _WindowRecorder:
    """Stand-in ``FallModelProtocol`` that records every window it is asked
    to score and returns an inert, never-qualifying probability. Used only to
    harvest real production-shaped windows for training; it never gates or
    emits anything itself.
    """

    def __init__(self) -> None:
        self.windows: list[tuple[tuple[float, ...], ...]] = []

    def predict(self, features):
        self.windows.append(tuple(tuple(float(v) for v in row) for row in features))
        return FallProbabilities(background=1.0, fall_transition=0.0, fallen=0.0)

    def warmup(self) -> None:
        return None


def _rel_time_s(pts_ns: int, t0_ns: int) -> float:
    return (pts_ns - t0_ns) / 1_000_000_000


def _owner_fall_windows(filename: str) -> list[tuple[float, float]]:
    if filename != OWNER_FALL_FILE:
        return []
    return [(center - half, center + half) for center, half in OWNER_FALL_WINDOWS_S]


def _subtract_intervals(
    total: tuple[float, float], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [total]
    for hole_start, hole_end in holes:
        next_pieces = []
        for start, end in pieces:
            if hole_end <= start or hole_start >= end:
                next_pieces.append((start, end))
                continue
            if hole_start > start:
                next_pieces.append((start, hole_start))
            if hole_end < end:
                next_pieces.append((hole_end, end))
        pieces = next_pieces
    return pieces


_SITE_POLICY = make_effective_policy(
    module_id="fall",
    module_version=2,
    values=FallPolicyV2(transition_threshold=0.5),
    source="image-default",
    facility_revision_id=None,
    camera_revision_id=None,
)


def harvest_site_negatives(path: Path) -> tuple[list[tuple[tuple[float, ...], ...]], dict]:
    raw = path.read_bytes()
    sha256 = sha256_bytes(raw)
    _, rows = decode_jsonl(raw.decode("utf-8"))
    t0_ns = rows[0].pts_ns
    duration_s = _rel_time_s(rows[-1].pts_ns, t0_ns)
    held_out_start_s = duration_s * (1.0 - HELD_OUT_FRACTION)
    exclusions = _owner_fall_windows(path.name)
    train_segments = _subtract_intervals((0.0, held_out_start_s), exclusions)

    recorder = _WindowRecorder()
    camera_id = rows[0].camera_id
    for seg_start, seg_end in train_segments:
        segment_rows = [r for r in rows if seg_start <= _rel_time_s(r.pts_ns, t0_ns) < seg_end]
        if len(segment_rows) < WINDOW:
            continue
        replay(
            camera_id=camera_id,
            rows=segment_rows,
            module_id="fall",
            policy=_SITE_POLICY,
            fall_model=recorder,
        )

    meta = {
        "file": path.name,
        "sha256": sha256,
        "duration_s": duration_s,
        "held_out_start_s": held_out_start_s,
        "train_segments_s": train_segments,
        "excluded_windows_s": exclusions,
        "n_train_windows": len(recorder.windows),
    }
    return recorder.windows, meta


# --------------------------------------------------------------------------
# Eval + provenance helpers
# --------------------------------------------------------------------------


def eval_report(scores: np.ndarray, labels: np.ndarray, thresholds: tuple[float, ...]) -> dict:
    if len(labels) == 0:
        return {"ap": 0.0, "n": 0, "n_positive": 0, "per_threshold": {}}
    order = np.argsort(-scores)
    ranked = labels[order]
    precision_at_k = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    ap = float((precision_at_k @ ranked) / max(int(ranked.sum()), 1))
    per_threshold = {}
    for t in thresholds:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        per_threshold[str(t)] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "promoted_count": int(pred.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return {
        "ap": ap,
        "n": len(labels),
        "n_positive": int(labels.sum()),
        "per_threshold": per_threshold,
    }


def _git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> None:
    print(f"[1/5] Loading AI-Hub parquet (revision {HF_REVISION})...", flush=True)
    import pyarrow.parquet as pq  # noqa: PLC0415 -- ephemeral training-only dependency

    parquet_path = _clip_parquet_path()
    payload_sha256 = sha256_bytes(parquet_path.read_bytes())
    table = pq.read_table(
        parquet_path,
        columns=["labels", "split_membership", "pose", "pose_head_bbox", "width", "height"],
    )
    by_split: dict[str, list[dict]] = {}
    for clip in table.to_pylist():
        by_split.setdefault(clip["split_membership"]["split_role"], []).append(clip)

    print("[2/5] Building AI-Hub train windows (sealed_test is held out, untouched)...", flush=True)
    x_train_aihub, y_train_aihub = _load_aihub_split(by_split, "train")
    x_sealed, y_sealed = _load_aihub_split(by_split, "sealed_test")
    print(
        f"  train: {len(y_train_aihub)} windows {dict(Counter(y_train_aihub.tolist()))}",
        flush=True,
    )
    print(f"  sealed_test: {len(y_sealed)} windows {dict(Counter(y_sealed.tolist()))}", flush=True)

    print("[3/5] Harvesting site-trace negatives via the real replay pipeline...", flush=True)
    site_meta = []
    site_features = []
    for trace_path in sorted(TRACES_DIR.glob("*.jsonl")):
        windows, meta = harvest_site_negatives(trace_path)
        site_meta.append(meta)
        site_features.extend(geometry_features(w, ASPECT) for w in windows)
        print(f"  {trace_path.name}: {meta['n_train_windows']} negative windows", flush=True)

    x_site = (
        np.array(site_features, dtype=np.float32)
        if site_features
        else np.zeros((0, GEOMETRY_FEATURE_DIM), dtype=np.float32)
    )
    y_site = np.zeros(len(x_site), dtype=np.int64)

    x_train = np.concatenate([x_train_aihub, x_site], axis=0)
    y_train = np.concatenate([y_train_aihub, y_site], axis=0)
    print(f"[4/5] Training RandomForestClassifier on {len(y_train)} windows "
          f"{dict(Counter(y_train.tolist()))}...", flush=True)

    model = RandomForestClassifier(**RF_PARAMS).fit(x_train, y_train)

    sealed_scores = model.predict_proba(x_sealed)[:, 1] if len(x_sealed) else np.zeros(0)
    sealed_eval = eval_report(sealed_scores, y_sealed, thresholds=(0.2, 0.5, 0.7))
    print(f"  sealed_test eval: {json.dumps(sealed_eval, indent=2)}", flush=True)

    print("[5/5] Exporting to ONNX and writing the artifact + receipt...", flush=True)
    import skl2onnx  # noqa: PLC0415 -- ephemeral training-only dependency
    from skl2onnx import convert_sklearn  # noqa: PLC0415
    from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415

    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, GEOMETRY_FEATURE_DIM]))],
        options={id(model): {"zipmap": False}},
        target_opset=17,
    )
    onnx_bytes = onnx_model.SerializeToString()
    onnx_sha256 = sha256_bytes(onnx_bytes)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "model.onnx").write_bytes(onnx_bytes)

    receipt = {
        "model_sha256": onnx_sha256,
        "threshold": PLACEHOLDER_THRESHOLD,
        "promotion_eligible": True,
        "feature_version": FEATURE_VERSION,
        "unapplied_policy_threshold": UNAPPLIED_POLICY_THRESHOLD,
        "dataset": {
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "clips_parquet_sha256": payload_sha256,
            "split_used_for_training": "train",
            "split_used_for_eval": "sealed_test",
        },
        "site_traces": site_meta,
        "owner_fall_excluded_from_training": {
            "file": OWNER_FALL_FILE,
            "windows_s_relative_to_t0": list(OWNER_FALL_WINDOWS_S),
        },
        "hyperparameters": {"algorithm": "RandomForestClassifier", **RF_PARAMS},
        "seed": SEED,
        "feature_dim": GEOMETRY_FEATURE_DIM,
        "aspect_ratio": ASPECT,
        "window_stride_guard": {"window": WINDOW, "stride": STRIDE, "guard": GUARD},
        "training_commit_sha": _git_head_sha(),
        "library_versions": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "skl2onnx": skl2onnx.__version__,
        },
        "sealed_aihub_eval": sealed_eval,
        "n_train_windows": {
            "aihub": len(y_train_aihub),
            "site_negatives": len(y_site),
            "total": len(y_train),
        },
    }
    (ARTIFACT_DIR / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {ARTIFACT_DIR / 'model.onnx'} (sha256={onnx_sha256})")
    print(f"Wrote {ARTIFACT_DIR / 'receipt.json'}")


if __name__ == "__main__":
    main()
