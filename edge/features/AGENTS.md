# edge/features — pure feature-math

L0 pure transforms for geometry, pose normalization, and sliding-window feature
extraction. Edge-internal (moved here from the top-level `features/` package in
the 3-instance refactor; consumed only by `edge/perception/`).

## Local Ownership
- `geometry.py`: IoU and greedy box matching.
- `pose_normalization.py`: per-person keypoint normalization.
- `window_features.py`: fall-window feature vectors and threshold constants.

## Imports
Allowed: standard library, numerical libraries, `contracts`, and local
`edge.features` modules.

Forbidden (enforced by import-linter): `backend`, `shared.events`, and any other
`edge` subpackage (`sources`, `perception`, `domains`, `runtime`, `evidence`,
`serving_client`, `runners`); filesystem reads, camera/video I/O, model loading.

## Gotchas
Historically vendored byte-identical with `eldercare-dataset-ops` (ADR-0004). Now
that it lives under `edge/`, it is no longer part of this repo's top-level
vendor-drift firewall (`tests/test_vendor_drift.py` covers `contracts` only) — if
dataset-ops still mirrors this feature-math, that sync is now manual/cross-repo.
`window_features._D = 45` is the single source of truth for the feature dimension —
do not derive it elsewhere; hardcode `45` as a literal with a comment if a consumer
needs the constant (matches `edge/perception/fall_window_classifier.py` and
`tests/test_features_window.py`).

## Focused Tests
- `tests/test_features_window.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary
- Keep transforms deterministic and free of filesystem or network side effects.
- Preserve `_D` as the feature-vector dimension source of truth.
