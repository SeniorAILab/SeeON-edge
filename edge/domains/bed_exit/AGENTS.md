# BED-EXIT DOMAIN KNOWLEDGE BASE

Own bed-exit event detection, bed occupancy schema, and per-person exit latching.

## Local Ownership

- `detector.py`: sticky own-bed assignment and exit event generation.
- `latch.py`: onset-only bed-exit latch.
- `schema.py`: bed status, frame, and event dataclasses.

## Imports

Allowed: `contracts`, `features` through `perception.tracker`, and local `domains.bed_exit`.

Forbidden: `edge/sources`, `edge/runners`, `edge` runtime orchestration, `shared.events`, `backend`, `training`.

## Focused Tests

- `tests/test_domains_bed_exit.py`
- `tests/test_demo_bed_exit.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

The detector uses person-box containment against bed boxes. Keep `hold_frames`, `grace_frames`, and `min_containment` behavior covered when changing assignment logic.
## Change Boundary

- Preserve own-bed assignment before interpreting an exit.
- Keep latching onset-only; downstream delivery is edge-owned.
- Test hold, grace, and containment thresholds together.
