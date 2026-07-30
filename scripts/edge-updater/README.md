# Edge updater daemon

`update-edge.sh` is a deterministic edge-host updater for `compose.edge.yaml`. It compares GHCR manifest digests with the running container image digests, snapshots the current edge inputs, pulls and applies the target images, verifies ml-api readiness + heartbeat + version digest, and reports every outcome locally and optionally to a backend URL.

## Required host inputs

Run from the repository root on the edge host.

- Docker with `docker compose` and either `docker buildx imagetools inspect` or `docker manifest inspect`.
- `.env.edge.prod` containing at least `ML_API_IMAGE`, `ML_WORKER_IMAGE`, and `ML_SERVING_PORT` as described by `.env.edge.prod.example`.
- `compose.edge.yaml`.
- Write access to `EDGE_UPDATER_DATA_DIR` (default `/var/lib/edge-updater`).

Useful env:

```sh
EDGE_UPDATER_DATA_DIR=/var/lib/edge-updater
EDGE_UPDATER_ENV_FILE=.env.edge.prod
EDGE_UPDATER_COMPOSE_FILE=compose.edge.yaml
EDGE_UPDATER_ML_API_BASE_URL=http://127.0.0.1:8000
EDGE_UPDATER_VERIFY_TIMEOUT_MIN=5
EDGE_UPDATER_REPORT_URL=https://backend.example.com/api/v1/edge-updater/report
```

`EDGE_UPDATER_REPORT_URL` is optional. Backend POST failures are not silent: the updater still appends to the local log and stores unsent JSON under `$EDGE_UPDATER_DATA_DIR/outbox`.

## Dry run

```sh
EDGE_UPDATER_DRY_RUN=1 scripts/edge-updater/update-edge.sh
```

Dry run performs `CHECK`, writes a local/report outcome, and stops before `SNAPSHOT`, `PULL`, `APPLY`, or env-file mutation.

## systemd timer example

`/etc/systemd/system/edge-updater.service`:

```ini
[Unit]
Description=Edge updater state machine
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/eldercare-fall-ml
Environment=EDGE_UPDATER_DATA_DIR=/var/lib/edge-updater
Environment=EDGE_UPDATER_VERIFY_TIMEOUT_MIN=5
EnvironmentFile=-/etc/edge-updater.env
ExecStart=/bin/sh /opt/eldercare-fall-ml/scripts/edge-updater/update-edge.sh
```

`/etc/systemd/system/edge-updater.timer`:

```ini
[Unit]
Description=Run edge updater periodically

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now edge-updater.timer
```

## launchd example

`/Library/LaunchDaemons/com.seniorailab.edge-updater.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.seniorailab.edge-updater</string>
  <key>WorkingDirectory</key><string>/opt/eldercare-fall-ml</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>/opt/eldercare-fall-ml/scripts/edge-updater/update-edge.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>EDGE_UPDATER_DATA_DIR</key><string>/var/lib/edge-updater</string>
    <key>EDGE_UPDATER_VERIFY_TIMEOUT_MIN</key><string>5</string>
  </dict>
  <key>StartInterval</key><integer>600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/var/log/edge-updater.log</string>
  <key>StandardErrorPath</key><string>/var/log/edge-updater.err</string>
</dict>
</plist>
```

## Local dry-run tests

```sh
sh scripts/edge-updater/test.sh
```

The test prepends mock `docker` and `curl` commands to `PATH` and verifies both the success path and the rollback path without contacting Docker, GHCR, ml-api, or the backend.
