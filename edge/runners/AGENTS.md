# RUNNERS KNOWLEDGE BASE

Own edge model runner adapters, model registry wiring, device selection, and warmup.

## Local Ownership

- `registry.py`: task name to runner factory mapping.
- `sklearn_fall.py`: trained fall model artifact loader and predictor.
- `yolo_pose.py`, `yolo_bed_seg.py`: YOLO runner adapters.
- `device.py`, `warmup.py`: runtime device choice and model warmup.

## Imports

Allowed: `contracts`, local `edge/runners`, numerical/model libraries, and standard library.

Forbidden: `edge/sources`, `edge/perception`, `edge/domains`, `edge` runtime orchestration, `shared.events`, `backend`, `training`.

## Focused Tests

- `tests/test_runners_registry.py`
- `tests/test_serving_model.py`
- `tests/test_demo_bed_detector.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

Runner swaps should stay behind `ModelRegistry`. Do not make callers import concrete runner classes unless a test or adapter needs explicit construction.
## Change Boundary

- Register new model adapters through `ModelRegistry` and preserve device/warmup ownership here.
