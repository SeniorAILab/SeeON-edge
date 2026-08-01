# Runbook: worker migration rollback (image digest)

Use this when a newly deployed `ml-worker` image misbehaves on an edge host and
you need the previous known-good worker back.

Rollback is **image-digest based, not a source revert**. Do not `git revert`,
rebuild, or edit Python on the edge host: the host pulls prebuilt GHCR images
only. You change one value — `ML_WORKER_IMAGE` — back to the prior digest and
restart that one service. Model, state, and clip volumes are preserved unchanged.

Inputs, exactly as described by
[`.env.edge.prod.example`](../../.env.edge.prod.example) and
[`compose.edge.yaml`](../../compose.edge.yaml):

```sh
cd /opt/eldercare-fall-ml   # the checkout that owns .env.edge.prod
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
```

`compose.edge.yaml` requires `ML_WORKER_PROFILE` (`cuda|mps|cpu`). If it is
missing from `.env.edge.prod`, even `$DC ps` fails with
`${ML_WORKER_PROFILE:?set cuda|mps|cpu}`. Fix that before any compose action, or
you will misread a config error as a rollback failure.

## 1. Record the digest you are leaving and the digest you are returning to

```sh
# Currently running worker image (resolved digest, not the mutable tag)
docker inspect --format '{{index .Config.Image}} {{.Image}}' \
  "$($DC ps -q ml-worker)"
docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
  ghcr.io/seniorailab/eldercare-fall-ml/ml-worker

# Previous edge-updater snapshot, if the updater performed the upgrade
sudo ls -1t /var/lib/edge-updater/snapshots | head -5
sudo grep -h ML_WORKER_IMAGE /var/lib/edge-updater/snapshots/<snapshot>/.env.edge.prod
```

The edge updater (`scripts/edge-updater/update-edge.sh`, documented in
[`scripts/edge-updater/README.md`](../../scripts/edge-updater/README.md))
snapshots the edge inputs before it applies images, so the prior
`ML_WORKER_IMAGE` digest is normally already on disk. If no snapshot exists, take
the digest from the `edge-ml-image-refs-<sha>` workflow artifact of the previous
release.

Write both digests down before continuing. Rolling back to a mutable tag defeats
the purpose; always use the `@sha256:<digest>` form.

## 2. Preserve the volumes (read this before you touch compose)

The worker's durable state is not in the image. `compose.edge.yaml` mounts:

| Mount | Kind | Contents |
| --- | --- | --- |
| `ml-worker-state:/var/lib/ml-worker` | named volume | `ML_WORKER_STATE_DIR`: LKG config, first-fault records, GPU lease, outbox DB |
| `ml-api-state:/var/lib/ml-api` | named volume | ml-api local state |
| `${CLIP_STORE_HOST_DIR}:/var/lib/clip-store` | host bind | evidence clips and manifests (worker `rw`, api `ro`) |
| `${ML_MODELS_DIR:-./models}:/app/models:ro` | host bind | model artifacts, read-only |

Rules:

- **Never** run `docker compose down -v` / `docker volume rm` during a rollback.
  `-v` destroys `ml-worker-state` and `ml-api-state`, taking the last-known-good
  config, the durable evidence outbox, and the first-fault record with it.
- Recreating the container is fine and expected; recreating the **volume** is not.
- The clip store and models directory are host paths. Leave
  `CLIP_STORE_HOST_DIR` and `ML_MODELS_DIR` untouched so both image versions read
  the same evidence and the same weights.
- Take a copy of the state volume before rolling back if the failing image may
  have migrated the on-disk outbox schema forward. A forward-migrated SQLite
  outbox is not guaranteed readable by the older image:

  ```sh
  sudo tar -C /var/lib/docker/volumes/eldercare-fall-ml_ml-worker-state/_data \
    -cf /var/tmp/ml-worker-state-pre-rollback.tar .
  ```

  Keep that archive until the rollback is verified. It is the only way back if
  the older image rejects a migrated outbox.

## 3. Set the prior digest and restart only the worker

Edit `.env.edge.prod` (gitignored; never commit it):

```sh
# before
ML_WORKER_IMAGE=ghcr.io/seniorailab/eldercare-fall-ml/ml-worker@sha256:<bad-digest>
# after
ML_WORKER_IMAGE=ghcr.io/seniorailab/eldercare-fall-ml/ml-worker@sha256:<prior-digest>
```

Leave every other key exactly as it is. The deployment identity is frozen: the
service and image stay `ml-worker`, the file names stay `compose.edge.yaml` /
`.env.edge.prod`, and the `ML_WORKER_*`, `WORKER_*`, `ML_API_*`, `API_*` keys are
unchanged across both image versions. A rollback that also renames something is
not a rollback.

```sh
$DC pull ml-worker
$DC up -d --no-deps ml-worker
```

`--no-deps` keeps `ml-api` running. The worker→ml-api relay surface
(`/api/v1/relay/{config,restart,alerts,heartbeat,runtime-status}`) is unchanged
by the worker migration, so the worker can be rolled back on its own. Roll back
`ML_API_IMAGE` only if ml-api itself is the faulty component.

## 4. Verify

```sh
# The container now runs the intended digest
docker inspect --format '{{.Image}}' "$($DC ps -q ml-worker)"

# ml-api stayed healthy and is receiving worker facts
$DC ps
curl -fsS http://127.0.0.1:"${ML_SERVING_PORT:-8000}"/health/ready

# Worker boot: profile resolution, model warmup, per-camera activation
$DC logs --tail 200 ml-worker
```

Expected: the digest matches the prior digest from step 1, ml-api is `healthy`,
the worker logs a resolved profile and completed warmup, and heartbeats resume.
Volumes must be the same ones as before — confirm with `docker volume ls` that
no `*_ml-worker-state` volume was recreated.

If the older image fails to start against the current state volume, restore the
step-2 archive into the volume and retry. If it fails again, the fault is not in
the worker image and this runbook is finished; escalate with the worker logs and
the first-fault record from `ML_WORKER_STATE_DIR`.

## 5. Close the loop

- Record both digests, the reason, and the outcome in the operations log.
- Keep the pre-rollback state archive until the replacement image ships.
- Report the failing digest against the migration issue so the next image is not
  built on the same defect. A rollback is not a fix.

## Related

- [`docs/architecture.md`](../architecture.md) — worker layers, entrypoint
  (`python -m worker`), and the failure matrix that tells you whether a fault is
  global (process exits non-zero) or per camera (only that camera degrades).
- [`docs/runbooks/driver-cuda-alignment.md`](driver-cuda-alignment.md) — host
  driver/CUDA faults. An image rollback does not repair a GPU in a bad state.
