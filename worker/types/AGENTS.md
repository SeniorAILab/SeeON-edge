# worker/types — internal envelopes

Own the worker's internal pipeline vocabulary: the small frozen dataclasses that
move between layers.

## Ownership rule

**`worker.types` imports only the standard library and `contracts`.** No
concrete I/O, model, framework, or higher-worker-layer import may be added here —
not now and not as a convenience later. This is the bottom of the ladder;
everything else imports it.

Enforced by import-linter contract *"worker.types imports only contracts from the
internal graph"*, which forbids `backend`, `shared`, `worker.interfaces`,
`worker.adapters`, `worker.pipeline`, `worker.domains`, and `worker.runtime`.

## Local Ownership

- `frame_packet.py`: `FramePacket(camera_id, frame, pts, seq, width, height,
  decode_time_ms)` — the only envelope allowed to carry an image.
- `module_result.py`: `ModuleResult(module_name, result, elapsed_ms)` wrapping a
  per-model `contracts.runner.RunnerResult`.
- `decision_input.py`: `DecisionInput` with exactly the seven legacy
  `DomainInput` fields; numeric only, never an image.
- `business_event.py`: `BusinessEvent(domain, event_type, identity, camera_id,
  facility_id, time_sec, probability, person_id?, bed_id?)`.

## Conventions

- `@dataclass(frozen=True, slots=True)` with `from __future__ import annotations`
  and an explicit `__all__`.
- `contracts.frame.Frame`, `contracts.runner.RunnerResult`,
  `contracts.observation.FrameObservation`, `BedRegionDebugSnapshot`, and
  `contracts.event.EventPayload` stay authoritative. Do not duplicate or shadow a
  vendored contract type, and do not add a class named `DetectionResult`.
- `Frame.image` is a mutable, unhashable NumPy array, so envelope hashes exclude
  authoritative payload fields and `FramePacket` excludes `frame` from value
  comparison. Preserve that when adding a field.

## Focused Tests

- `tests/test_worker_types.py`
- `tests/test_import_dependency_ladder.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

Prefer a new envelope over widening an existing one. An envelope that would need
an I/O or model import belongs in `worker/interfaces` (as a port) or
`worker/adapters` (as an implementation detail), not here.
