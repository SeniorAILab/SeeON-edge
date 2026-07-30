# EVENTS KNOWLEDGE BASE

Own L4 outbound alert/event shape, publishers, outbox, and backend Event API client.

## Local Ownership

- `schemas.py`: emitted event and backend Event API payload construction.
- `local_publisher.py`: network-free publisher protocol and logging/stub implementation.
- `outbox.py`: publisher-backed outbox.
- `edge_ingest_client.py`: no-HMAC backend Event API alert and heartbeat HTTP client using the single `API_BACKEND_EVENTS_URL` base.

## Imports

Allowed: `contracts` and local `shared.events`.

Forbidden: `backend`, `edge`, `training`.

## Focused Tests

- `tests/test_events_schema.py`
- `tests/test_events_outbox.py`
- `tests/test_events_ingest_client.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

The backend egress contract is the no-HMAC Event API: post events to the single configured `API_BACKEND_EVENTS_URL` and heartbeats to `API_BACKEND_EVENTS_URL + "/heartbeat"` with JSON bodies only.
## Delivery Boundary

- Backend egress remains in `edge_ingest_client.py`; publishers and outboxes do not open cameras or load models.
