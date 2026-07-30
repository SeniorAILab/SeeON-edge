# WORKER DOMAINS KNOWLEDGE BASE

Own edge domain detectors and the `DOMAIN_REGISTRY`.

## Local Ownership

- `base.py`: `DomainDetector` interface.
- `__init__.py`: `DomainRegistration`, enabled-domain helpers, and registry wiring.
- `fall/`, `bed_exit/`: enabled domain implementations.

## Imports

Allowed: `contracts`, `edge.features`, `edge/perception`, and local `edge/domains`.

Forbidden: `edge/sources`, `edge/runners`, `edge` runtime orchestration, `shared.events`, `backend`, `training`.

## Focused Tests

- `tests/test_domains_fall.py`
- `tests/test_domains_bed_exit.py`
- `tests/test_domain_registry_scaffolds_disabled.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

`edge/domains` must not import edge runtime orchestration. Edge runtime owns scheduling and camera identity; domains own observation-to-event interpretation only.
## Registry Rule

- Add an enabled detector through `DOMAIN_REGISTRY` with its input view, event types, debug adapter, and audit metadata provider.
- The registry derives production domain configuration and relay event allowlists. Add a domain there; do not recreate parallel runtime allowlists.
- **Registry-only scope, stated precisely.** A *judgment* module that consumes the existing shared perception input (`DomainInput`) is fully registry-only: register it in `DOMAIN_REGISTRY`, enable it in `domains.enabled`, and its events reach the relay with no runtime, config, or relay edits.
- **Not yet registry-only:** a domain needing a *new model or runner* still requires runtime and config changes. `_WorkerResources.fall_model`, `_build_fall_model`, `WorkerModelsConfig.fall`, and the `FallModelProtocol` factory argument assume a single fall model. Generalising model provisioning is deliberately out of scope for now — do not describe the registry as covering it.
