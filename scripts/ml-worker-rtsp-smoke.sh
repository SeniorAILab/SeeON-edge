#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'MSG'
Usage: EDGE_CAMERA_CONFIG=/path/to/ml-worker.yaml scripts/ml-worker-rtsp-smoke.sh

Probes every RTSP URL in the ml-worker YAML config, then runs ml-worker for a
bounded number of frames per camera.

Common env:
  EDGE_CAMERA_CONFIG      Required ml-worker YAML config
  MAX_FRAMES_PER_CAMERA   Frames per camera before exit, default: 30
MSG
}

config="${EDGE_CAMERA_CONFIG:-}"
frames="${MAX_FRAMES_PER_CAMERA:-30}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$config" == "-h" || "$config" == "--help" ]]; then
  usage
  exit 2
fi
if [[ -z "$config" ]]; then
  printf 'EDGE_CAMERA_CONFIG is required\n' >&2
  usage
  exit 2
fi
if [[ ! -f "$config" ]]; then
  printf 'EDGE_CAMERA_CONFIG does not exist: %s\n' "$config" >&2
  exit 2
fi

config_path="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"

urls_output="$(
  uv run python - "$config_path" <<'PY'
import sys
from edge.runtime.edge_worker_config import EdgeWorkerConfigError, load_edge_worker_config

try:
    config = load_edge_worker_config(sys.argv[1])
except EdgeWorkerConfigError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc

for camera in config.cameras:
    print(camera.inference_rtsp_url)
PY
)"
mapfile -t urls <<<"$urls_output"

if [[ "${#urls[@]}" -lt 1 ]]; then
  printf 'expected at least one camera in %s\n' "$config_path" >&2
  exit 2
fi

for url in "${urls[@]}"; do
  ffprobe \
    -v error \
    -rtsp_transport tcp \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height \
    -of default=noprint_wrappers=1 \
    "$url" >/dev/null
done

uv run python -m edge.runtime.edge_worker \
  --config "$config_path" \
  --max-frames-per-camera "$frames" \
  --heartbeat-on-start
