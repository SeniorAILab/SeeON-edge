#!/usr/bin/env bash
# scripts/research/fall-deploy.sh -- E2 candidate deploy for the SeeON fall v1
# research stack. Resolves a version/alias from the MLflow registry, downloads
# the HF bundle, smoke-tests it twice (edge venv + in-image ORT-CPU), then
# atomically swaps it in and restarts ml-worker.
#
# Usage:
#   fall-deploy.sh <alias|version>       # e.g. fall-deploy.sh candidate
#   fall-deploy.sh --check-repo <repo>   # standalone hf_repo allowlist check
set -euo pipefail

MLFLOW_URL="http://100.96.162.127:15000"
MODEL_NAME="seeon-fall"
REPO_RE='^Berom0227/'
HF="${HF_BIN:-$HOME/.local/bin/hf}"
die() { echo "abort: $*" >&2; exit 1; }
check_repo() { [[ "$1" =~ $REPO_RE ]]; }

if [[ "${1:-}" == "--check-repo" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: fall-deploy.sh --check-repo <repo>" >&2; exit 2; }
  check_repo "$2" && { echo "ok: $2 matches $REPO_RE"; exit 0; }
  echo "reject: $2 does not match $REPO_RE" >&2
  exit 1
fi

REF="${1:?usage: fall-deploy.sh <alias|version>}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E="${HOME}/seeon-research/edge"
DC=(docker compose -p seeon-beomsu-edge --env-file "$E/research.env" -f "$R/scripts/research/compose.research.yaml")

# 1. Resolve hf_repo/hf_revision/onnx_sha256 from the MLflow registry.
if [[ "$REF" =~ ^[0-9]+$ ]]; then
  ENDPOINT="$MLFLOW_URL/api/2.0/mlflow/model-versions/get?name=$MODEL_NAME&version=$REF"
else
  ENDPOINT="$MLFLOW_URL/api/2.0/mlflow/registered-models/alias?name=$MODEL_NAME&alias=$REF"
fi
MV_JSON="$(curl -sf "$ENDPOINT")"
read -r VERSION HF_REPO HF_REVISION ONNX_SHA256 <<<"$(python3 -c '
import json, sys
mv = json.loads(sys.argv[1])["model_version"]
tags = {t["key"]: t["value"] for t in mv.get("tags", [])}
print(mv["version"], tags["hf_repo"], tags["hf_revision"], tags["onnx_sha256"])
' "$MV_JSON")"
check_repo "$HF_REPO" || die "hf_repo '$HF_REPO' does not match $REPO_RE"
echo "resolved seeon-fall v$VERSION -> $HF_REPO@$HF_REVISION (onnx $ONNX_SHA256)"

# 2. Download the bundle, skipped when present; both paths verify model.onnx below.
B="$E/models/fall/bundles/seeon-fall-$VERSION"
B_IN="/app/models/fall/bundles/seeon-fall-$VERSION"
if [[ -d "$B" ]]; then
  echo "bundle already present: $B"
else
  "$HF" auth whoami >/dev/null 2>&1 || die "hf not logged in: run hf auth login"
  "$HF" download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$B"
fi
echo "$ONNX_SHA256  $B/model.onnx" | sha256sum -c - >/dev/null \
  || die "bundle $B has a model.onnx that does not match onnx_sha256 $ONNX_SHA256"
echo "bundle verified: $B (onnx sha256 matches)"

# 3. Bundle smoke: edge venv, then ORT-CPU in-image; both must print the same digest line, or abort before swap.
SMOKE_PY='
from worker.runtime.config.local_env import fall_model_config_from_environment as cfg
from worker.adapters.model.ort_pose_bbox56 import load_packaged_fall_bundle as load
b = load(cfg().artifact_dir)
print("digest", b.published_weights_digest, "model.pt", "tau", b.runner.receipt_threshold, "m", b.runner.receipt_transition_votes)
'
smoke_env_args() {
  local dir="$1"
  printf '%s\n' \
    "ML_WORKER_PROFILE=flow" \
    "ML_WORKER_FALL_MODEL_ARTIFACT_DIR=$dir" \
    "ML_WORKER_FALL_MODEL_WINDOW=30" \
    "ML_WORKER_FALL_MODEL_STRIDE=5" \
    "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD=0.5"
}
mapfile -t EDGE_ENV < <(smoke_env_args "$B")
EDGE_OUT="$(cd "$R" && env "${EDGE_ENV[@]}" PYTHONPATH="$R" "$R/.venv/bin/python" -c "$SMOKE_PY")"
echo "edge venv: $EDGE_OUT"
mapfile -t IMG_ENV < <(smoke_env_args "$B_IN")
IMG_ARGS=()
for kv in "${IMG_ENV[@]}"; do IMG_ARGS+=(-e "$kv"); done
IMG_OUT="$("${DC[@]}" run --rm --no-deps --entrypoint python "${IMG_ARGS[@]}" ml-worker -c "$SMOKE_PY")"
echo "in-image: $IMG_OUT"
[[ "$EDGE_OUT" == "$IMG_OUT" ]] || die "smoke mismatch: '$EDGE_OUT' != '$IMG_OUT'"

# 4. Atomic symlink swap, shared by the forward swap and rollback; ln -sfn tmp link survives an aborted run's leftover.
swap_symlink() {
  (cd "$E/models" && ln -sfn "$1" fall/.pose-bbox56-gru.tmp \
    && mv -T fall/.pose-bbox56-gru.tmp fall/pose-bbox56-gru)
}
PREV_TARGET="$(readlink "$E/models/fall/pose-bbox56-gru" 2>/dev/null || true)"
swap_symlink "bundles/seeon-fall-$VERSION"

# 5. Recreate ml-worker, poll up to 60s for it to boot (restart: "no" means up -d alone proves nothing); roll back on failure.
wait_for_boot() {
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if [[ -n "$("${DC[@]}" ps -q --status running ml-worker)" ]] \
      && "${DC[@]}" logs ml-worker 2>&1 | grep -q 'fall model operating_threshold resolved to'; then
      return 0
    fi
    sleep 2
  done
  return 1
}
T0="$(date +%s)"
if ! "${DC[@]}" up -d --no-deps --force-recreate ml-worker || ! wait_for_boot; then
  echo "abort: ml-worker did not boot within 60s; rolling back" >&2
  if [[ -n "$PREV_TARGET" ]]; then
    swap_symlink "$PREV_TARGET"
    "${DC[@]}" up -d --no-deps --force-recreate ml-worker || true
  fi
  exit 1
fi

# 6. Record the deploy.
read -r _ DIGEST _ _ TAU _ M <<<"$EDGE_OUT"
mkdir -p "$HOME/seeon-research/runs"
echo "$T0 v$VERSION onnx=$ONNX_SHA256 digest=$DIGEST tau=$TAU m=$M" >>"$HOME/seeon-research/runs/fall-deploy.log"
echo "deployed seeon-fall v$VERSION at $T0"
