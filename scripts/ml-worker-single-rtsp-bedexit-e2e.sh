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

# Sandbox worker runtime state (GPU lease file, durable evidence outbox) under
# this run's disposable tmpdir instead of the real /var/lib/ml-worker, and
# turn on relay export -- EvidenceExportRuntime.from_environment() (worker/
# pipeline/output/evidence/evidence_runtime.py) is disabled unless
# ML_WORKER_EVENT_CLIP_EXPORT_ENABLED=1, in which case admitted events stage
# to the durable outbox but are never sent to the relay, and this harness's
# whole point is proving relay delivery.
export ML_WORKER_STATE_DIR="$tmpdir/worker-state"
export ML_WORKER_EVIDENCE_OUTBOX_PATH="$tmpdir/worker-state/evidence-outbox.sqlite3"
export ML_WORKER_EVENT_CLIP_EXPORT_ENABLED=1
mkdir -p "$ML_WORKER_STATE_DIR"

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

# Event admission (worker/domains/bed_exit) and relay export (worker/pipeline/
# output/evidence/evidence_runtime.py's EvidenceExportRuntime sender) run on
# separate background threads inside the worker process, so the alert can
# land in the backend a short interval after run_worker_with_clock returns.
# Retry the exact same readback query instead of a single point-in-time
# check.
wait_for_alert_count_at_least() {
  local started_at="$1" minimum="$2" count=0
  for _ in $(seq 1 20); do
    count="$(alert_count_since "$started_at")"
    if [[ "$count" -ge "$minimum" ]]; then
      printf '%s\n' "$count"
      return 0
    fi
    sleep 1
  done
  printf '%s\n' "$count"
  return 1
}

run_worker_with_clock() {
  local now="$1"
  if ! OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" BED_EXIT_NOW="$now" EDGE_CAMERA_CONFIG="$config" uv run python - "$frames" >>"$worker_log" 2>&1 <<'PY'
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, final

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult, bed_result, person_result, pose_result
from worker.runtime.config import load_worker_config
from worker.runtime.worker import WorkerRuntime


@final
class _EmptyPoseRunner:
    """No pose data: bed-exit tracking runs on person geometry alone."""

    def __call__(self, _image: Image) -> RunnerResult:
        return pose_result(poses=(), boxes=())

    def warmup(self) -> None:
        return None


# Same calibrated geometry as tests/e2e_worker_relay_fixtures.py's
# ScriptedBedExitPersonRunner, against the REAL (unconfigurable-via-YAML)
# BedExitConfig defaults worker/runtime/worker.py._bed_exit_config always
# applies: min_containment=0.35, hold_frames=2, grace_frames=3. Steps 1-2
# fully contain the person (assigns the bed by step 2); steps 3-4 partially
# overlap (still >=0.35, resets grace); step 5 onward is fully outside (grace
# increments each frame, firing once grace_frames > 3, i.e. on the 4th
# consecutive frame outside -- step 8). Matches this script's default
# MAX_FRAMES_PER_CAMERA=8; raising the frame count only holds the final
# outside position longer and does not change when the event fires.
_BED_EXIT_XS: Final = (15.0, 15.0, 40.0, 65.0, 90.0, 90.0, 90.0, 90.0)
_BED_BOX: Final = (10.0, 10.0, 90.0, 80.0, 0.9)


@final
class _WalkingPersonRunner:
    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def __call__(self, _image: Image) -> RunnerResult:
        with self._lock:
            index = min(self._index, len(_BED_EXIT_XS) - 1)
            self._index += 1
        x1 = _BED_EXIT_XS[index]
        return person_result(boxes=((x1, 15.0, x1 + 70.0, 75.0, 0.95),))

    def warmup(self) -> None:
        return None


@final
class _FixedBedRunner:
    def __call__(self, _image: Image) -> RunnerResult:
        return bed_result(boxes=(_BED_BOX,))

    def warmup(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _InertFallModel:
    """Never consulted -- this run's config enables bed_exit only -- but
    WorkerRuntime._initialize_models() always calls serving.create("fall")
    at boot regardless of enabled domains, so a well-formed stand-in is
    still required."""

    def __init__(self) -> None:
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def predict(self, _features: object) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _CountingPersonRunner:
    """Wraps the person runner to signal once MAX_FRAMES_PER_CAMERA decoded
    frames have been pulled through the real per-frame pipeline (person is
    scheduled every frame; bed is only rescanned every 30 frames, so person
    is the correct per-frame proxy)."""

    def __init__(self, inner: _WalkingPersonRunner, *, target: int, done: threading.Event) -> None:
        self._inner = inner
        self._target = target
        self._done = done
        self._count = 0
        self._lock = threading.Lock()

    def __call__(self, image: Image) -> RunnerResult:
        result = self._inner(image)
        with self._lock:
            self._count += 1
            if self._count >= self._target:
                self._done.set()
        return result

    def warmup(self) -> None:
        self._inner.warmup()


@final
class _ScriptedServingClient:
    def __init__(self, *, target_frames: int, done: threading.Event) -> None:
        self._runners: dict[str, object] = {
            "pose": _EmptyPoseRunner(),
            "person": _CountingPersonRunner(
                _WalkingPersonRunner(), target=target_frames, done=done
            ),
            "bed": _FixedBedRunner(),
            "fall": _InertFallModel(),
        }

    def create(self, task: str, **_options: object) -> object:
        try:
            return self._runners[task]
        except KeyError as error:
            raise AssertionError(f"unexpected serving task requested: {task!r}") from error


class _FrozenDatetime(datetime):
    """Freezes worker.runtime.worker's bed-exit night-window clock.

    worker/runtime/worker.py:_build_decider hardcodes
    `clock=lambda: datetime.now(UTC)` for the bed_exit decider with no
    injection seam (it is the ONLY `datetime` use in that whole module), and
    worker.domains.registry.DOMAIN_REGISTRY is an immutable
    types.MappingProxyType, so the legacy edge_worker approach of swapping in
    a replacement DomainRegistration no longer works. Instead, monkeypatch
    the bare module-level `datetime` name worker.runtime.worker resolves at
    call time -- the same idiom tests/e2e_worker_relay_fixtures.py already
    uses for worker_module.ClipRecorderConfig. No production file is edited.
    """

    _frozen: datetime

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        return cls._frozen if tz is None else cls._frozen.astimezone(tz)


def _wait_until(predicate, *, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def main() -> int:
    frames = int(sys.argv[1])
    frozen_now = datetime.fromisoformat(os.environ["BED_EXIT_NOW"])
    if frozen_now.tzinfo is None:
        print("BED_EXIT_NOW must carry a UTC offset", file=sys.stderr)
        return 1
    _FrozenDatetime._frozen = frozen_now
    worker_module.datetime = _FrozenDatetime

    config = load_worker_config(os.environ["EDGE_CAMERA_CONFIG"])
    done = threading.Event()
    serving = _ScriptedServingClient(target_frames=frames, done=done)
    runtime = WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        hard_exit=lambda _code: None,
    )

    thread = threading.Thread(target=runtime.run, daemon=True, name="single-rtsp-bedexit-worker")
    thread.start()
    try:
        if not _wait_until(lambda: len(runtime.cameras) > 0, timeout=30.0):
            print("camera did not activate within 30s", file=sys.stderr)
            return 1
        if not done.wait(timeout=max(60.0, frames * 5.0)):
            print(f"did not observe {frames} decoded frames within timeout", file=sys.stderr)
            return 1
        # Event admission -> durable staging -> the async export sender's
        # relay POST (worker/pipeline/output/evidence/evidence_runtime.py)
        # run on background threads, not synchronously with frame
        # processing -- give them a settle window before tearing down.
        time.sleep(5.0)
    finally:
        runtime.stop()
        thread.join(timeout=15.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
if ! night_count="$(wait_for_alert_count_at_least "$night_started_utc" 1)"; then
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
