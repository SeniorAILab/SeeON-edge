# FALL DOMAIN KNOWLEDGE BASE

Own fall-event latching and fall event schema.

## Local Ownership

- `detector.py`: `FallEventLatch`, rising-edge detection, and observation-to-event dict conversion.
- `schema.py`: `FallEvent` dataclass.

## Imports

Allowed: `contracts` and local `domains.fall`.

Forbidden: `edge/perception`, `edge` runtime orchestration, `shared.events`, `backend`, `training`, model runners, sources.

## Focused Tests

- `tests/test_domains_fall.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

`FallEventLatch.update` is the edge-facing `FrameObservation` -> event-payload API. Use `update_signal` for low-level boolean rising-edge tests and status latching.
## Change Boundary

- Preserve rising-edge semantics at the detector boundary.
- Keep event schema changes additive for edge consumers.
- Use `FrameObservation` at the edge-facing API.
- Keep low-level signal tests on `update_signal`.
- Cover latch changes in `tests/test_domains_fall.py`.
