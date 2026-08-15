# Edge updater daemon

`update-edge.sh` is a deterministic edge-host updater for `compose.edge.yaml`.
It compares GHCR manifest digests with the running container image digests,
snapshots the current edge inputs, pulls and applies the target images,
verifies ml-api readiness + heartbeat + version digest, and reports every
outcome locally and optionally to a backend URL.

The supported Linux deployment path is a **root-owned systemd oneshot/timer**.
That carrier owns the canonical checkout, the private env file, updater
state, and host-derived VAAPI GID bindings. It does **not** confer Docker
root-equivalence on an interactive deploy user. Containers remain root as
designed. Do not add the operator to the Docker group, mount the Docker
socket into a container, or run a privileged updater container.

## Required host inputs

Run from the sealed checkout on the edge host, or via the systemd unit after
the host operator has installed the carrier.

- Docker with `docker compose` and either `docker buildx imagetools inspect`
  or `docker manifest inspect`.
- A mode-`0600` private env file containing at least the digest-pinned
  `ML_API_IMAGE` and `ML_WORKER_IMAGE` values described by
  `.env.edge.prod.example`. Intel VAAPI also requires `EDGE_RENDER_GID` and
  `EDGE_VIDEO_GID` copied from the live host; there is no repository default.
- `compose.edge.yaml` (plus `compose.edge.igpu.yaml` or
  `compose.edge.migrate.yaml` when that host uses those overlays).
- A carrier-writable deploy root and a root-owned
  `EDGE_UPDATER_DATA_DIR` (default `/var/lib/edge-updater`).

Useful env, normally supplied by the audited binding file rather than an
interactive shell:

```sh
EDGE_UPDATER_DATA_DIR=/var/lib/edge-updater
EDGE_UPDATER_ENV_FILE=.env.edge.prod
EDGE_UPDATER_COMPOSE_FILE=compose.edge.yaml
EDGE_UPDATER_ML_API_BASE_URL=http://127.0.0.1:8000
EDGE_UPDATER_VERIFY_TIMEOUT_MIN=5
EDGE_UPDATER_REPORT_URL=https://backend.example.com/api/v1/edge-updater/report
```

`EDGE_UPDATER_REPORT_URL` is optional. Backend POST failures are not silent:
the updater still appends to the local log and stores unsent JSON under
`$EDGE_UPDATER_DATA_DIR/outbox`.

Set `EDGE_HOST_PREFLIGHT=1` for the fail-closed host binding checks: root
carrier identity, writable deploy root, mode-0600 env, updater-state
ownership, digest-pinned image references, and a clean worktree when the
deploy root is a git checkout. Process environment must not redirect the
executable; the systemd unit pins `ExecStart`.

## Dry run

```sh
EDGE_UPDATER_DRY_RUN=1 scripts/edge-updater/update-edge.sh
```

Dry run performs `CHECK`, writes a local/report outcome, and stops before
`SNAPSHOT`, `PULL`, `APPLY`, or env-file mutation.

## systemd carrier (supported Linux path)

Committed templates live in `scripts/edge-updater/systemd/`. They are
examples, not a host checkout. Initial unit installation is a host-operator
action:

1. Copy the sealed `update-edge.sh` to
   `/usr/local/libexec/seeon-edge/update-edge.sh` (root-owned, not writable
   by an interactive deploy user).
2. Create the audited binding file `/etc/seeon/edge-deploy.env` mode `0600`.
   It names the deploy root, private env, compose files, updater state, and
   host GIDs. It is not optional.
3. Install `seeon-edge-updater.service` and `seeon-edge-updater.timer`.
4. `systemctl daemon-reload && systemctl enable --now seeon-edge-updater.timer`

`seeon-edge-updater.service`:

```ini
[Unit]
Description=SeeON edge updater
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
User=root
Group=root
EnvironmentFile=/etc/seeon/edge-deploy.env
Environment=EDGE_HOST_PREFLIGHT=1
ExecStart=/bin/sh /usr/local/libexec/seeon-edge/update-edge.sh
```

`seeon-edge-updater.timer`:

```ini
[Unit]
Description=Run SeeON edge updater periodically

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min
Persistent=true
Unit=seeon-edge-updater.service

[Install]
WantedBy=timers.target
```

The binding file, not the unit, selects the checkout. Do not hardcode a
facility path in the committed unit. `Environment=` must not change
`ExecStart`.

Example `/etc/seeon/edge-deploy.env` keys (values are host-local):

```sh
EDGE_HOST_PREFLIGHT=1
EDGE_CARRIER_UID=0
EDGE_DEPLOY_ROOT=/srv/seeon/edge
EDGE_UPDATER_DATA_DIR=/var/lib/edge-updater
EDGE_UPDATER_ENV_FILE=/srv/seeon/edge/.env.edge.prod
EDGE_UPDATER_COMPOSE_FILE=/srv/seeon/edge/compose.edge.yaml
EDGE_UPDATER_VERIFY_TIMEOUT_MIN=5
```

## Fresh vs cutover preflight

Both modes require the root-owned carrier, a clean sealed checkout, mode-0600
env, digest-pinned images, and idle updater state.

Greenfield additionally requires:

- base `compose.edge.yaml` only (no migrate overlay)
- importer `--fresh-install`
- no `EDGE_LEGACY_*` volume names

Cutover additionally requires:

- `compose.edge.migrate.yaml`
- `EDGE_LEGACY_CATALOG_VOLUME`, `EDGE_LEGACY_CONNECTION_VOLUME`,
  `EDGE_LEGACY_WORKER_VOLUME`
- importer `--require-sources catalog,connection,worker`

Intel VAAPI additionally requires host-derived `EDGE_RENDER_GID` and
`EDGE_VIDEO_GID` that match `/dev/dri/renderD128`.

## launchd example

`/Library/LaunchDaemons/com.seniorailab.edge-updater.plist` remains a
macOS development convenience only. Production Linux uses the systemd
carrier above.

## Local dry-run tests

```sh
sh scripts/edge-updater/test.sh
```

The test prepends mock `docker` and `curl` commands to `PATH` and verifies
both the success path and the rollback path without contacting Docker, GHCR,
ml-api, or the backend.
