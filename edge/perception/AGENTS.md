# PERCEPTION KNOWLEDGE BASE

Own edge observation construction and scene state derived from frames and runner outputs.

## Local Ownership

- `observation_builder.py`: converts raw detections, poses, and bed boxes into `FrameObservation`.
- `tracker.py`: greedy IoU tracking.
- `window_buffer.py`: temporal frame windows.
- `scene_state.py`: scene-level state.

## Imports

Allowed: `contracts`, `edge.features`, and local `edge/perception`.

Forbidden: `edge/sources`, `edge/runners`, `edge/domains`, `edge` runtime orchestration, `shared.events`, `backend`, `training`.

## Focused Tests

- `tests/test_perception_observation_builder.py`
- `tests/test_frame_observation_contract.py`
- `tests/test_demo_tracking.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

Perception builds facts about a frame; it does not decide whether an alert should fire. Put domain-specific latches under `edge/domains`.
## Change Boundary

- Keep perception outputs observational; domain detectors own alert semantics and latching.
