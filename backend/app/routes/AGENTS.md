# API ROUTES KNOWLEDGE BASE

Own app-level FastAPI route modules (health, models). Business logic stays in `backend.app.lifespan`, feature slices, or explicit gateway helpers.

## Local Ownership

- `health.py`: live, ready, and legacy health responses.
- `status.py`: runtime/status-store snapshot.
- `models.py`: gateway metadata; no model registry or loaded-model state.

## Imports

Allowed: `backend.app`, lower-layer read-only facades needed by a route, FastAPI, Pydantic.

Forbidden: `training`, `edge`, direct camera opening, direct model training.

## Focused Tests

- `tests/test_serving_health.py`
- `tests/test_serving_status.py`
- `tests/test_serving_models.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

Route validation is part of the public API. Keep error status codes covered when changing relay/status/models request validation or exception mapping.
## Change Boundary

- Keep product routes under `/api/v1`; keep health probes unversioned.
- Handlers validate and delegate; lifespan owns collaborator construction.
