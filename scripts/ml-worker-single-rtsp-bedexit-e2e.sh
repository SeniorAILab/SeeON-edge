#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_base_url="${BACKEND_BASE_URL:-http://127.0.0.1:8080}"
relay_base_url="${RELAY_URL:-http://127.0.0.1:8000}"
relay_token="${RELAY_TOKEN:-local-edge-relay-token}"
facility_id="${E2E_FACILITY_ID:?E2E_FACILITY_ID is required}"
resident_id="${E2E_RESIDENT_ID:?E2E_RESIDENT_ID is required}"
camera_id="${E2E_CAMERA_ID:?E2E_CAMERA_ID is required}"
night_now="${BED_EXIT_NIGHT_NOW:-2026-06-25T22:00:00+09:00}"
day_now="${BED_EXIT_DAY_NOW:-2026-06-25T13:00:00+09:00}"
night_window_start="${BED_EXIT_NIGHT_WINDOW_START:-21:00}"
night_window_end="${BED_EXIT_NIGHT_WINDOW_END:-05:00}"
night_window_tz="${BED_EXIT_NIGHT_WINDOW_TZ:-Asia/Seoul}"
frames="${MAX_FRAMES_PER_CAMERA:-8}"
rtsp_url="${BED_EXIT_RTSP_URL:?BED_EXIT_RTSP_URL is required; start SeniorAILab/rtsp-generator separately and pass a worker-reachable URL}"
compose_project="${COMPOSE_PROJECT_NAME:-ml-worker-single-rtsp-bedexit-e2e}"
db_container="${E2E_DB_CONTAINER:-eldercare-fall-db}"
postgres_user="${POSTGRES_USER:-fall}"
postgres_db="${POSTGRES_DB:-fall_dev}"
tmp_root="${ML_EDGE_E2E_TMP_ROOT:-$repo_root/.gjc/tmp}"
mkdir -p "$tmp_root"
tmpdir="$(mktemp -d "$tmp_root/ml-worker-single-bedexit.XXXXXX")"
config="$tmpdir/ml-worker.yaml"
api_log="$tmpdir/ml-api.log"
worker_log="$tmpdir/ml-worker.log"
api_pid=""

cleanup() {
  if [[ -n "$api_pid" ]]; then
    kill "$api_pid" >/dev/null 2>&1 || true
    wait "$api_pid" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmpdir"
}

require_backend() {
  if ! curl -fsS "${backend_base_url}/" >/dev/null 2>&1; then
    printf 'backend is not running or /health is unavailable: %s\n' "$backend_base_url" >&2
    printf 'start backend first (for example: pnpm dev:backend) and retry.\n' >&2
    return 1
  fi
}

write_config() {
  cat >"$config" <<YAML
version: 1
relay:
  url: ${relay_base_url}
  token: ${relay_token}
runtime:
  max_failures: 30
  open_timeout_ms: 20000
  read_timeout_ms: 20000
domains:
  bed_exit:
    enabled: true
    night_window:
      start: "${night_window_start}"
      end: "${night_window_end}"
      tz: ${night_window_tz}
cameras:
  - camera_id: ${camera_id}
    facility_id: ${facility_id}
    resident_id: ${resident_id}
    rtsp_url: ${rtsp_url}
    heartbeat_interval_sec: 30
    frame_stride: 1
    label: single-bed-exit
YAML
  chmod 600 "$config"
}

start_api() {
  API_BACKEND_EVENTS_URL="${backend_base_url}/api/v1/events" \
  API_EDGE_RELAY_TOKEN="$relay_token" \
  API_CAMERA_INVENTORY="[{\"camera_id\":\"${camera_id}\",\"facility_id\":\"${facility_id}\",\"resident_id\":\"${resident_id}\"}]" \
  uv run uvicorn backend.app.main:app --host 127.0.0.1 --port "${relay_base_url##*:}" >"$api_log" 2>&1 &
  api_pid="$!"
  for _ in $(seq 1 60); do
    if curl -fsS "${relay_base_url}/health/live" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf 'ml-api did not become healthy; log follows:\n' >&2
  sed -n '1,120p' "$api_log" >&2
  return 1
}

alert_count_since() {
  local started_at="$1"
  docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc \
    "select count(*) from alerts where facility_id = '${facility_id}' and type = 'bed-exit' and detected_at >= '${started_at}'::timestamptz;" | tr -d '[:space:]'
}

run_worker_with_clock() {
  local now="$1"
  if ! OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" BED_EXIT_NOW="$now" EDGE_CAMERA_CONFIG="$config" uv run python - "$frames" >>"$worker_log" 2>&1 <<'PY'
from __future__ import annotations

import os
import sys
from datetime import datetime

from worker.domains import DOMAIN_REGISTRY, DomainRegistration
from worker.domains.bed_exit.detector import BedExitMonitor, NightWindow
from worker import edge_worker


class ScriptedPoseRunner:
    def predict(self, _frame):
        return ((), ())


class ScriptedPersonRunner:
    def predict(self, _frame):
        if not hasattr(self, "calls"):
            self.calls = 0
        self.calls += 1
        if self.calls <= 2:
            return ((10, 10, 90, 80, 0.98),)
        return ((52, 10, 132, 80, 0.98),)


class ScriptedBedRunner:
    def predict(self, _frame):
        return ((0, 0, 90, 100, 0.99),)


class ScriptedFallRunner:
    operating_threshold = 0.5

    class metadata:
        window = 3
        stride = 1

    def predict(self, _features):
        return 0.0


class ScriptedRegistry:
    def create(self, name: str, **kwargs):
        if name == "pose":
            return ScriptedPoseRunner()
        if name == "person":
            return ScriptedPersonRunner()
        if name == "bed":
            return ScriptedBedRunner()
        if name == "fall":
            return ScriptedFallRunner()
        raise KeyError(name)


def bed_exit_factory(config):
    raw = config["night_window"] if isinstance(config, dict) else {}
    return BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(
            start=str(raw.get("start", "21:00")),
            end=str(raw.get("end", "05:00")),
            tz=str(raw.get("tz", "Asia/Seoul")),
        ),
        clock=lambda: datetime.fromisoformat(os.environ["BED_EXIT_NOW"]),
    )


DOMAIN_REGISTRY["bed_exit"] = DomainRegistration("bed_exit", bed_exit_factory, True)
edge_worker.DOMAIN_REGISTRY["bed_exit"] = DOMAIN_REGISTRY["bed_exit"]
edge_worker.DEFAULT_REGISTRY = ScriptedRegistry()
raise SystemExit(edge_worker.main(["--config", os.environ["EDGE_CAMERA_CONFIG"], "--max-frames-per-camera", sys.argv[1]]))
PY
  then
    printf 'worker run failed; log follows:\n' >&2
    sed -n '1,200p' "$worker_log" >&2
    return 1
  fi
}

trap cleanup EXIT
require_backend
write_config
start_api
printf 'clock night=%s day=%s night_window=%s-%s %s relay=%s backend_events=%s rtsp=%s\n' \
  "$night_now" "$day_now" "$night_window_start" "$night_window_end" "$night_window_tz" \
  "$relay_base_url" "$backend_base_url" "$rtsp_url"
night_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker_with_clock "$night_now"
night_count="$(alert_count_since "$night_started_utc")"
if [[ "$night_count" -lt 1 ]]; then
  printf 'night bed-exit did not reach backend Event API; worker log follows:\n' >&2
  sed -n '1,160p' "$worker_log" >&2
  printf '%s\n' '--- ml-api log ---' >&2
  sed -n '1,200p' "$api_log" >&2
  exit 1
fi
printf 'night alert count: %s\n' "$night_count"

day_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker_with_clock "$day_now"
day_count="$(alert_count_since "$day_started_utc")"
if [[ "$day_count" -ne 0 ]]; then
  printf 'daytime bed-exit was not suppressed; new backend alert count=%s\n' "$day_count" >&2
  exit 1
fi
printf 'day suppress count: %s\n' "$day_count"
printf 'single RTSP bed-exit relay harness ok: camera=%s facility=%s\n' "$camera_id" "$facility_id"
