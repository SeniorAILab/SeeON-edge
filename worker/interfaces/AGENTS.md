# worker/interfaces — replaceable seams

Own one `typing.Protocol` per swappable seam. This package is the contract layer
that lets a decoder, model, encoder, decider, or sink be replaced without editing
a pipeline caller.

## Ownership rule

**`worker.interfaces` imports only the standard library, `contracts`, and
`worker.types`.** No concrete logic, no I/O, no model or framework import, no
default implementation. A Protocol body is `...`; if you find yourself writing
behavior here, it belongs in `worker/adapters` or `worker/pipeline`.

Enforced by import-linter contracts *"worker types and interfaces dependency
ladder"* and *"worker.interfaces imports only contracts and worker.types from the
internal graph"*, which forbid `backend`, `shared`, `worker.adapters`,
`worker.pipeline`, `worker.domains`, and `worker.runtime`.

## Local Ownership

- `decode.py`: `DecodeAdapter.open(config) -> DecodeSession`; `DecodeSession`
  exposes `read() -> FramePacket | None` and `close()`.
- `bus.py`: named, bounded, per-camera frame-bus publish/subscribe surface.
- `extract.py`: `Extractor.extract(FramePacket) -> ModuleResult`.
- `decision.py`: `Decider.update(DecisionInput) -> tuple[BusinessEvent, ...]`.
- `encode.py`: `ClipEncoder.open(camera, profile, geometry) -> EncoderSession`
  and `ClipFinalizer.finalize(segments, event) -> artifact`.
- `output.py`: `EventSink.emit(BusinessEvent)`.
- `serving.py`: `ServingClient.create(task, ...)` plus the typed but deliberately
  unimplemented `BatchServingClient.infer_batch` (ADR-0002 deferral).

## Conventions

- `@runtime_checkable` where a test asserts substitutability.
- Name a port for the capability, not the vendor: `DecodeAdapter`, not
  `OpenCVAdapter`.
- Every seam has at least two implementations, or one implementation plus a test
  double, before it is considered proven.
- The `ClipEncoder` input/output contract must stay permissive enough for a
  future packet-copy adapter (ADR-0001 follow-up) without changing any decision
  or output caller.

## Focused Tests

- `tests/test_worker_interfaces.py`
- `tests/test_import_dependency_ladder.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

Adding a seam is cheap; changing an existing port's signature is not. Prefer an
additional Protocol over a breaking signature change, and never let a port leak a
concrete type (an FFmpeg process, a cv2 capture, a torch module) into its
signature.
