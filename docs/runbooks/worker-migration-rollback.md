# Migrate or roll back edge state safely

The current edge release has one writable SQLite database:
`/var/lib/seeon-state/edge.sqlite3` in the `edge-state` volume. API and worker
remain separate HTTP-connected runtimes, but both validate that central schema
and write only their owned table families. They never execute DDL.

The stopped-runtime `edge-db-migrator` is the only DDL owner. It checkpoints,
backs up, integrity-checks, and imports the released `catalog.sqlite3`,
`connection-settings.sqlite3`, and `worker-state.sqlite3`. A legacy worker
source at schema 6 or 7 is migrated through outbox schema 8 in a secured working
copy before its rows are imported; schema 8 and 9 sources import unchanged. The
original legacy database is not upgraded.

Rollback is digest-based and decided per database file and traffic boundary.
Never open any database with a binary whose maximum supported schema is lower
than that database's schema. Never use a mutable image tag, direct SQL, or an
environment-derived facility identity.

## Before migration

Record the exact running and target API/worker image digests and retain the sealed
previous Compose deployment artifact. Stop API and worker, acquire the edge
deployment lock, and reject a running updater.

For each legacy database independently:

1. record its schema version
2. checkpoint WAL with `PRAGMA wal_checkpoint(TRUNCATE)`
3. use the SQLite online backup API into a root-only `0700` directory and `0600`
   timestamped file
4. run `PRAGMA integrity_check`
5. fsync the backup file and parent directory
6. record only filename, SHA-256, size, schema, table counts, and row counts

Require healthy volumes, drained queues, at least 20 GiB free clip capacity, and
verified backups before changing containers. Retain the untouched
`ml-api-state` and `ml-worker-state` volumes throughout the rollback window.
They are mounted only by the stopped-runtime migrator during import; the API and
worker must never mount or write them after cutover.

## Migrate and cut over

Obtain the deployment command once:

```sh
cd /opt/eldercare-fall-ml
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
```

Run the one-shot import. This is the only command allowed to execute DDL:

```sh
$DC up --pull always --no-deps edge-db-migrator
```

Stop unless it exits zero and reports `EDGE_DB_IMPORT_OK`. Confirm its source
receipts, counts, digests, and integrity result before admitting traffic. Then
preserve the required dependency order:

```sh
$DC up -d --wait ml-api
$DC up -d --wait ml-worker
```

Compose enforces `edge-db-migrator` completed successfully -> `ml-api` healthy ->
`ml-worker`. Do not start either runtime around the dependency gates. Record
the first successful API or worker write to `edge.sqlite3` as the **central cutover traffic boundary**. From that point, legacy snapshots are stale.

## Rollback decision

### Before central cutover traffic

If the migrator or readiness checks fail before any API or worker process writes
central state, a rollback to the untouched legacy volumes is permitted. Decide
compatibility separately for `catalog.sqlite3`,
`connection-settings.sqlite3`, and `worker-state.sqlite3`; every previous binary
must support the exact schema of the database it will open. Restore the sealed
previous Compose artifact and recorded image digests, remount the untouched
legacy volumes, and start the previous API before the previous worker.

Do not restore a backup over an untouched legacy volume. Do not mix a central
runtime with a legacy runtime, and do not copy imported central rows back into a
legacy database.

### After central cutover traffic

Once central traffic exists, never restore or restart a legacy database: doing so
would discard enrollment, topology, heartbeat, outbox, config, or fault writes
made after import. Preserve `edge-state` and fix forward, or roll back only to
recorded API/worker digests that explicitly support the current central schema.
A binary-only rollback must not restore database bytes.

For a central-compatible binary rollback:

1. stop new work and drain relay/evidence queues
2. verify both previous images support the current `edge.sqlite3` schema
3. restore the recorded immutable image digests
4. recreate `ml-api` and wait for `/health/ready`
5. recreate `ml-worker` only after the API is healthy
6. verify `/api/v1/system`, normal heartbeat, topology, and event idempotency

Never use `down -v`, delete `edge-state`, edit SQLite manually, or promise rollback
to an outbox schema-6/7 binary after it would open schema-8-compatible state.

## Roll forward

Deploy the sealed central-schema-compatible digests through the same migrator ->
API healthy -> worker sequence. On an already imported database the migrator is
idempotent and verifies the recorded source digests/counts rather than duplicating
rows. Enrollment and camera topology remain database/UI/Hub-owned; do not stamp a
facility ID into worker config or reconstruct it from a backup filename.

Run the lifecycle smoke only after the content-addressed execution receipt is
green. Boolean declarations are not lifecycle evidence: the receipt must contain
sequenced successful executions for enrollment, topology, heartbeat, event/clip,
rotation, timeout retry, ML-first rollback, roll-forward, and restart.

```sh
sh scripts/ops/cloud-enrollment-smoke.sh \
  --host happy-nursing-home-raw --full-lifecycle --rollback-drill
```

## Cutover mapping and backend delivery gates

Do not accept a real event, clip transfer, or Vercel read-side result until the
reusable delivery gate is green. Expected camera count and the designated real
witness are operator-private snapshot facts, not a repository roster and not a
host-specific room label.

Obtain a sealed pre-cutover snapshot receipt and an independently recorded
SHA-256. The snapshot lists opaque local camera IDs and exactly one designated
witness from that same list. Then collect a redacted delivery readout with
enrollment, per-camera Backend mappings, per-camera external heartbeat results,
authenticated SSE timing, and the witness event/clip/Vercel proof.

```sh
export EDGE_PROVISIONING_SNAPSHOT=/secure/pointers/pre-cutover-snapshot.json
export EDGE_PROVISIONING_SNAPSHOT_SHA256='<independently recorded snapshot digest>'
export EDGE_PROVISIONING_DELIVERY=/secure/pointers/cutover-delivery-readout.json
export EDGE_PROVISIONING_DELIVERY_SHA256='<independently recorded readout digest>'
sh scripts/ops/cloud-enrollment-smoke.sh --delivery-gate \
  --snapshot "$EDGE_PROVISIONING_SNAPSHOT" \
  --delivery "$EDGE_PROVISIONING_DELIVERY"
sh scripts/ops/cloud-enrollment-smoke.sh --print-checklist
```

Stop, and do not enable witness processing or treat an event as acceptance, unless
every item below holds:

- enrollment succeeded and was authenticated
- the snapshot content-address still matches the sealed digest
- the snapshot-derived expected camera count equals the number of listed cameras
- every expected camera has a non-empty Backend camera ID
- `mapping_pending` is false for every expected camera
- every expected camera has a successful, fresh external heartbeat
- authenticated Hub SSE is established before designated-witness processing
- the witness event is real, not fabricated
- the resulting clip and Vercel read-side proof are authenticated

A single unmapped camera, blank mapping, `mapping_pending=true`, missed
heartbeat, or stale heartbeat is a clear stop. Count drift between the sealed
snapshot and the readout is a clear stop. The gate prints only reason tokens,
counts, and SHA bindings. It must not print RTSP, tokens, cookies, or other
secrets, and it must not contact a production Hub.

Preserve the existing privilege and rollback boundary: no Docker-socket mount,
no privileged container, no Docker-group grant, and no legacy-volume restore
after central cutover traffic. Intel hosts still require `EDGE_RENDER_GID` to
equal the live `/dev/dri/renderD128` owner GID; there is no repository GID
default.

Success evidence contains only plan/SHA/digest bindings, backup metadata,
versioned API statuses, counts, and redacted lifecycle results. Never put SQLite
bytes or secret values in `.omo` evidence.
