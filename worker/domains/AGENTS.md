# worker/domains — event interpretation

Own the interpretation of numeric observations into business events, plus the
`DOMAIN_REGISTRY`. Fall and bed-exit are the enabled domains.

## Ownership rule

**`worker.domains` must not import `worker.pipeline.ingest`,
`worker.pipeline.output`, or `worker.runtime`.** A domain never opens a stream,
never sends a relay request, never writes a clip, and never learns how it was
scheduled. It receives `DecisionInput` and returns `BusinessEvent` values; the
pipeline decides what to do with them.

Allowed: `contracts`, `worker.types`, `worker.interfaces`, and
`worker.pipeline.perception` (numeric feature math and observation types).

Enforced by import-linter contracts *"worker domains do not depend on ingest,
output, or runtime"* and *"worker runtime is the sole composition root"*.

## Local Ownership

- `base.py`: the `Decider` implementation base and `DomainDetector` surface.
- `__init__.py`: `DomainRegistration`, enabled-domain helpers, `DOMAIN_REGISTRY`.
- `fall/`: window classifier, per-track windows, probability, rising-edge latch.
- `bed_exit/`: sticky own-bed assignment, containment, grace/hold counters, night
  window, debug snapshot.

## Conventions

- `Decider.update(DecisionInput) -> tuple[BusinessEvent, ...]`. Return typed
  events, never a dict and never a raw frame.
- No image access. If a decision needs pixel-derived information, extract it in
  `worker/pipeline/perception` and add it to the observation.
- Rising-edge only: repeated positive frames emit one event. Latching is onset
  semantics; downstream delivery is not a domain concern.
- Time comes from an injected, timezone-aware clock, so night-window and
  cross-midnight behavior is testable without wall time.
- Per camera: classifier buffers, per-track windows, probability, latch, bed
  assignments, grace/hold counters. Shared: the model object only.

## Registry Rule

- Add an enabled detector through `DOMAIN_REGISTRY` with its input view, event
  types, debug adapter, and audit metadata provider. The registry derives
  production domain configuration and the relay event allowlist — do not build a
  parallel allowlist.
- **Registry-only scope, stated precisely.** A judgment module that consumes the
  existing `DecisionInput` is fully registry-only: register it, enable it in
  `domains.enabled`, and its events reach the relay with no runtime, config, or
  relay edits.
- **Not registry-only:** a domain needing a new model or runner still requires
  runtime and config changes, because fall-model provisioning assumes a single
  fall model. Generalising model provisioning is deliberately out of scope — do
  not describe the registry as covering it.

## Focused Tests

- `tests/test_domains_fall.py`, `tests/test_domains_bed_exit.py`
- `tests/test_domain_registry_scaffolds_disabled.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

Keep event schema changes additive. Cover hold, grace, and containment thresholds
together, and prove per-camera isolation whenever detector state changes shape.
