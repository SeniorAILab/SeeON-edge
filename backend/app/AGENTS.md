# BACKEND INSTANCE KNOWLEDGE BASE

Own the FastAPI backend instance (`ml-api` image): app factory, lifespan boot,
the edge→backend `/api/v1/relay/*` gateway, backend Event API egress, gateway
metadata, and a relay-heartbeat-derived `/status`. The backend is the edge node's
single no-HMAC Event API gateway; it does not assemble live camera loops and
shares no in-memory state with the edge worker.

## Vertical-slice 2-depth ownership

- `main.py` — `create_app`, `/api/v1` router registration.
- `lifespan.py` — thin-gateway boot and `app.state` assembly.
- `core/` — settings/config.
- `features/<slice>/` — one capability per slice, each with `router.py` +
  `store.py` (+ schemas): `auth`, `cameras`, `clips`, `evidence`, `relay`, `status`
  (status owns the relay-heartbeat + runtime-status stores; clips owns clip
  store + label audit). `routes/` keeps app-level `health` + `models`.
- `shared/` — cross-cutting infra only (e.g. `backend_mapping`, dashboard sessions); never feature state.

Store ownership: `camera_registry`→cameras, `clip_store`→clips,
`runtime_status_store`+`heartbeat_store`→status; the relay slice consumes them via
FastAPI deps.

## Imports

Allowed: `contracts`, `shared.events.edge_ingest_client`, local `backend.app.*`,
FastAPI/Pydantic.

Forbidden (enforced by import-linter): `edge`; and base layers (`core`/`shared`)
must not import the upper feature/route/app layers.

## FastAPI / wire-schema convention

- HTTP wire schemas are Pydantic `BaseModel` (never `dataclass`) and live in the
  backend layer (the slice, or a shared `schemas.py` once reused). Never put wire
  schemas in `contracts`; L0 stays framework-free.
- Naming: request = `<Action>Request`, response = `<Action>Response`; a trivial ack
  may return a typed `dict[str, str]`.
- Strictness: request models set `model_config = ConfigDict(extra="forbid")` and
  validate every field with `Field(...)`; routes declare `response_model=...`.
  Header/Query deps use `typing.Annotated` (Ruff `FAST`).
- Settings: `pydantic_settings.BaseSettings` + `SettingsConfigDict(env_prefix="ML_API_", extra="ignore")`.
- Routers: one `APIRouter(prefix=..., tags=[...])` per slice with thin handlers;
  product routes under `/api/v1`, health probes unversioned; end with `__all__ = ["router"]`.
- Injected collaborators are `typing.Protocol` bound in `lifespan.py`.

## Focused Tests

- `tests/test_serving_api.py`, `tests/test_serving_health.py`,
  `tests/test_serving_status.py`, `tests/test_serving_models.py`
- `tests/test_serving_boundary_contract.py`, `tests/test_api_ingest_relay.py`,
  `tests/test_api_heartbeat_store.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

`lifespan.py` boots a thin gateway (config, Event API gateway, heartbeat store,
readiness) — it does NOT assemble camera loops, domain detectors, or edge runtime
state. `/api/v1/relay/heartbeat` stamps local `received_at` after auth + camera
binding and before backend egress so `/api/v1/status` reflects edge-local truth
even when backend egress fails. Keep handlers thin; product routes stay under
`/api/v1`, `/health/live` and `/health/ready` unversioned.
