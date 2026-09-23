"""Site-replay + sealed-AI-Hub validation of the trained fall-geometry classifier.

Reports exactly the three things the trained classifier's promotion depends
on, for both the gated and ungated variant of ``TrainedGeometryFallScorer``,
at three candidate operating thresholds (0.2, 0.5, 0.7):

(a) owner-fall recall on the one camera with real recorded falls
    (``d9c8ada04907a3e8.jsonl``, both t0-relative ±10s of 398s and 1695s);
(b) false-positive "fall" episodes per camera / camera-hour, both on each
    trace's held-out last-25% block and on the full trace;
(c) sealed AI-Hub ``sealed_test`` split AP/precision/recall/promoted-count.

Every replay call drives the real ``worker.replay.engine.replay`` pipeline
(same production code the training script harvested negatives through) with
a real ``TrainedGeometryFallScorer`` loaded from the artifact this repo's
training run produced (``models/fall/geometry-trained-v1/``, sha256-verified
on load same as a worker boot would). Thresholds are swept through the
policy layer (``FallPolicyV2.transition_threshold``), not the scorer, exactly
matching how an operator would configure it.

Run: PYTHONPATH=. uv run --with pyarrow python3 scripts/replay_validate_fall_geometry_classifier.py
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from contracts.replay_trace import decode_jsonl
from scripts.train_fall_geometry_classifier import (
    ARTIFACT_DIR,
    ASPECT,
    OWNER_FALL_FILE,
    OWNER_FALL_WINDOWS_S,
    TRACES_DIR,
    _clip_parquet_path,
    _clip_windows,
)
from shared.detection_policies import FallPolicyV2, make_effective_policy
from worker.domains.fall.geometry_features import collapse_gate
from worker.domains.fall.trained_scorer import TrainedGeometryFallScorer
from worker.interfaces.fall_model import FallProbabilities
from worker.replay.engine import replay

THRESHOLDS = (0.2, 0.5, 0.7)
GATED_VARIANTS = (True, False)
OWNER_FALL_TOLERANCE_S = 10.0
PR581_REPORTED_TIME_SEC = 2700.375


class _FlatModel:
    """Same packaged-proxy stand-in the geometry-scorer tests already use --
    0.05 is the packaged proxy's own recorded score on a real corridor fall
    (see worker/runtime/worker.py's comment). Shadow-only: never gates."""

    def predict(self, features):
        del features
        return FallProbabilities(0.95, 0.05, 0.0)

    def warmup(self) -> None:
        return None


def _site_replay_table() -> dict:
    receipt = json.loads((ARTIFACT_DIR / "receipt.json").read_text())
    by_file = {t["file"]: t for t in receipt["site_traces"]}

    results: dict[str, dict] = {}
    for trace_path in sorted(TRACES_DIR.glob("*.jsonl")):
        meta = by_file[trace_path.name]
        _, rows = decode_jsonl(trace_path.read_bytes().decode("utf-8"))
        t0_ns = rows[0].pts_ns
        camera_id = rows[0].camera_id
        frame_width, frame_height = rows[0].frame_width, rows[0].frame_height
        held_out_start_s = meta["held_out_start_s"]
        duration_s = meta["duration_s"]

        is_owner_file = trace_path.name == OWNER_FALL_FILE
        owner_windows = (
            [
                (c - OWNER_FALL_TOLERANCE_S, c + OWNER_FALL_TOLERANCE_S)
                for c, _ in OWNER_FALL_WINDOWS_S
            ]
            if is_owner_file
            else []
        )

        per_variant: dict[str, dict] = {}
        for gated in GATED_VARIANTS:
            scorer = TrainedGeometryFallScorer.from_artifact_dir(
                ARTIFACT_DIR,
                _FlatModel(),
                frame_width=frame_width,
                frame_height=frame_height,
                gated=gated,
            )
            per_threshold: dict[str, dict] = {}
            for threshold in THRESHOLDS:
                policy = make_effective_policy(
                    module_id="fall",
                    module_version=2,
                    values=FallPolicyV2(transition_threshold=threshold),
                    source="image-default",
                    facility_revision_id=None,
                    camera_revision_id=None,
                )
                run = replay(
                    camera_id=camera_id,
                    rows=rows,
                    module_id="fall",
                    policy=policy,
                    fall_model=scorer,
                )
                events = [e for frame in run.frames for e in frame.events if e.event_type == "fall"]
                rel_times = sorted((e.time_sec - t0_ns / 1_000_000_000) for e in events)

                recalled = [
                    any(lo <= t <= hi for t in rel_times) for lo, hi in owner_windows
                ]
                fps_full = [
                    t for t in rel_times if not any(lo <= t <= hi for lo, hi in owner_windows)
                ]
                fps_held = [t for t in fps_full if t >= held_out_start_s]
                held_out_hours = (duration_s - held_out_start_s) / 3600
                full_hours = duration_s / 3600

                fp_per_camera_hour_full = len(fps_full) / full_hours if full_hours else 0.0
                fp_per_camera_hour_held = len(fps_held) / held_out_hours if held_out_hours else 0.0
                per_threshold[str(threshold)] = {
                    "n_fall_events": len(events),
                    "event_times_s_relative": rel_times,
                    "owner_fall_recall": recalled if is_owner_file else None,
                    "fp_full_trace": len(fps_full),
                    "fp_full_trace_per_camera_hour": fp_per_camera_hour_full,
                    "fp_held_out_block": len(fps_held),
                    "fp_held_out_per_camera_hour": fp_per_camera_hour_held,
                }
            per_variant["gated" if gated else "ungated"] = per_threshold
        results[trace_path.name] = {"camera_id": camera_id, "duration_s": duration_s, **per_variant}
    return results


def _sealed_eval_table() -> dict:
    from worker.adapters.model.ort_vector_classifier import OrtVectorClassifier
    from worker.domains.fall.geometry_features import GEOMETRY_FEATURE_DIM, geometry_features

    receipt = json.loads((ARTIFACT_DIR / "receipt.json").read_text())
    classifier = OrtVectorClassifier.from_model_path(
        ARTIFACT_DIR / "model.onnx",
        feature_dim=GEOMETRY_FEATURE_DIM,
        expected_digest=receipt["model_sha256"],
    )

    parquet_path = _clip_parquet_path()
    table = pq.read_table(
        parquet_path,
        columns=["labels", "split_membership", "pose", "pose_head_bbox", "width", "height"],
    )
    sealed_clips = [
        c for c in table.to_pylist() if c["split_membership"]["split_role"] == "sealed_test"
    ]

    windows_labels: list[tuple[tuple, int]] = []
    for clip in sealed_clips:
        windows_labels.extend(_clip_windows(clip))

    results = {}
    for gated in GATED_VARIANTS:
        scores, labels = [], []
        for window, label in windows_labels:
            gate_ok = (not gated) or collapse_gate(window, ASPECT)
            score = (
                classifier.positive_probability(geometry_features(window, ASPECT))
                if gate_ok
                else 0.0
            )
            scores.append(score)
            labels.append(label)

        import numpy as np

        arr_scores = np.array(scores, dtype=np.float32)
        arr_labels = np.array(labels, dtype=np.int64)
        order = np.argsort(-arr_scores)
        ranked = arr_labels[order]
        precision_at_k = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
        ap = float((precision_at_k @ ranked) / max(int(ranked.sum()), 1))

        per_threshold = {}
        for t in THRESHOLDS:
            pred = arr_scores >= t
            tp = int((pred & (arr_labels == 1)).sum())
            fp = int((pred & (arr_labels == 0)).sum())
            fn = int((~pred & (arr_labels == 1)).sum())
            per_threshold[str(t)] = {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "promoted_count": int(pred.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        results["gated" if gated else "ungated"] = {
            "ap": ap,
            "n": len(arr_labels),
            "n_positive": int(arr_labels.sum()),
            "per_threshold": per_threshold,
        }
    return results


def main() -> None:
    t0_ns_d9c8ada = None
    for trace_path in sorted(TRACES_DIR.glob("*.jsonl")):
        if trace_path.name == OWNER_FALL_FILE:
            _, rows = decode_jsonl(trace_path.read_bytes().decode("utf-8"))
            t0_ns_d9c8ada = rows[0].pts_ns
            break
    t0_s = t0_ns_d9c8ada / 1_000_000_000
    pr581_rel_s = PR581_REPORTED_TIME_SEC - t0_s
    print(
        f"PR#581 time_sec={PR581_REPORTED_TIME_SEC} -> t0-relative={pr581_rel_s:.3f}s "
        f"(owner fall #1 at 398s, #2 at 1695s; within +/-10s of #2: "
        f"{abs(pr581_rel_s - 1695.0) <= 10.0})"
    )

    print("\n=== (a)+(b) site replay ===")
    site_results = _site_replay_table()
    print(json.dumps(site_results, indent=2))

    print("\n=== (c) sealed AI-Hub eval ===")
    sealed_results = _sealed_eval_table()
    print(json.dumps(sealed_results, indent=2))

    out_path = ARTIFACT_DIR / "validation_report.json"
    out_path.write_text(
        json.dumps(
            {
                "pr581_time_sec_conversion": {
                    "reported_time_sec": PR581_REPORTED_TIME_SEC,
                    "t0_relative_s": pr581_rel_s,
                    "matches_owner_fall_2": abs(pr581_rel_s - 1695.0) <= 10.0,
                },
                "site_replay": site_results,
                "sealed_aihub_eval": sealed_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
