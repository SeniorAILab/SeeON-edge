# EDGE INSTANCE KNOWLEDGE BASE

Own the deployable edge instance (`ml-worker` image): RTSP ingest + decode,
perception, domain interpretation, evidence egress, and **in-process model
inference behind the `serving_client` seam**. Reads camera config, builds shared
runners with one composition-root device selection, opens RTSP sources, runs
supervisors, creates alert/heartbeat facts with probability, and relays them to
the backend over one-way HTTP.

## Internal 2-depth ownership

- `sources/` — RTSP/webcam/video-file frame sources (NVDEC/OpenCV backends).
- `perception/` — tracker, observation builder, scene state, window buffer, fall
  window classifier.
- `domains/` — fall and bed-exit interpretation/latching.
- `runners/` — model adapters + `ModelRegistry`, device selection, warmup.
- `serving_client/` — `ServingClient` interface + `InProcessServingClient`
  (pass-through over `ModelRegistry`); the seam exposes a batch-input contract a
  future networked batched serving service swaps in without a rewrite (ADR-0002).
- `evidence/` — clip recorder + outbox/reconciliation/retention + sender.
- `runtime/` — `edge_worker` (CLI/composition root), `camera_worker`, supervisor,
  scheduler, config pull/resolve, status/latest-frame/incident flow state, mjpeg.
- `features/` — edge-owned pure feature-math (geometry, pose norm, window features).

Entry point: `python -m worker` (canonical worker CLI).

## Imports

Allowed: `contracts`, `shared.events`, and local `edge.*` modules.

Forbidden (enforced by import-linter): `backend`. Instances talk only over relay
HTTP. The internal 2-depth layer order (runtime→evidence→domains→perception→sources)
is a layers contract; `serving_client` is a low-level seam.

## Focused Tests

- `tests/test_worker_entrypoint.py`, `tests/test_worker_runner_sharing.py`
- `tests/test_edge_worker_supervisor.py`, `tests/test_edge_worker_four_streams.py`
- `tests/test_edge_serving_client.py` (seam equivalence)
- `tests/test_worker_backend_ingest_contract.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Gotchas

Runner objects are intentionally shared across camera workers — preserve object
identity when changing `_RunnerBundle` or supervisor construction. Provision
runners through `serving_client` (`InProcessServingClient`), not the registry
directly. Edge owns ALL live flow state (status/latest-frame/incident/detector
windows); there is no cross-process shared state with the backend. The only
edge↔backend connection is one-directional relay HTTP (`edge → backend
/api/v1/relay/*`); classification probability is produced here and relayed as
backend Event API `confidence`.
