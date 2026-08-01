# worker/pipeline — the five stages

Own the staged data path: ingest, frame bus, perception/analytics, decision
aggregation, and output. Stages hold per-camera state and call ports; they do not
own process lifecycle.

## Ownership rule

**`worker.pipeline` must not import `worker.runtime`.** The runtime composes and
injects; a stage that imports the composition root creates a cycle and makes the
stage untestable in isolation. Take config values, adapters, clocks, and sinks as
arguments.

`worker.pipeline.perception` is stricter still: it stays **pure numeric math**
and must not import `backend`, `shared.events`, `worker.adapters`,
`worker.pipeline.{bus,ingest,decision,output}`, `worker.domains`, or
`worker.runtime`. This keeps observation building and feature math testable
without a camera, a model, or a filesystem.

Enforced by import-linter contracts *"worker runtime is the sole composition
root"* and *"worker perception features stay pure"*.

## Local Ownership

- `ingest/`: source registry and descriptor validation, per-camera capture loop,
  reconnect, pacing, RTSP credential masking. Imports the `DecodeAdapter` port,
  never a concrete decoder.
- `bus/`: named bounded per-camera subscriptions — `inference` latest-only
  capacity 1, `live` latest-only capacity 1, `evidence` bounded FIFO (default
  128) — plus `latest_frame.py` and `scheduler.py`. Publish/take/drop counters and
  queue age live here.
- `perception/`: observation builder, greedy tracker, `SceneState`, window
  buffer, `DecisionInput` construction, and `features/` (geometry, pose
  normalization, window features).
- `decision/`: `incident_manager.py` — cooldown keys, event admission, persisted
  event identity, camera/facility/time enrichment.
- `output/`: `EventSink` relay egress, `evidence/` (segment encoder use, clip
  finalizer, manifests, outbox, retention, reconciliation), `overlay.py`,
  `live_view.py`, `snapshot_store.py`.
- `camera_pipeline.py`: per-camera orchestration only — wiring, no business math.

## Conventions

- `FramePacket` is published immutably. Any consumer that draws or mutates copies
  the image first.
- Raw frames go only to model extraction, derivative evidence, overlay/live view,
  and the alert snapshot. The decision stage receives `DecisionInput`.
- No unbounded queue anywhere. A full subscription drops by its documented policy
  and increments its counter exactly once.
- Source construction or iteration failure means camera `DEGRADED` plus
  `camera.offline`. A later processing failure is **not** an offline source
  failure — keep the categories distinct.
- Clip/snapshot/relay side effects happen after event admission, in `output/`
  only.
- Keep new pure-code modules at or below 250 logical LOC; split by stage rather
  than recreating a monolith.

## Focused Tests

- `tests/test_pipeline_bootstrap.py`
- `tests/test_import_dependency_ladder.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

Per-camera state stays per camera. Never hoist a tracker, `SceneState`, window
buffer, scheduler, `IncidentManager`, or encoder ring into module or process
scope to "save memory" — only model objects are shared.
