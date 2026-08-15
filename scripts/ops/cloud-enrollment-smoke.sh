#!/usr/bin/env sh
# allow: SIZE_OK - rollout ordering and fail-closed transitions remain auditable in one script.
set -eu
set +x

MIN_EDGE_CLIP_MIB=20480
MODE=
DRY_RUN=false
DEPLOY=false
RESTART_CHECK=false
FULL_LIFECYCLE=false
ROLLBACK_DRILL=false
DELIVERY_GATE=false
PRINT_CHECKLIST=false
HOST=
PLAN=${EDGE_PROVISIONING_PLAN:-}
DRAFT=${EDGE_PROVISIONING_DRAFT:-}
SEAL=${EDGE_PROVISIONING_SEAL:-}
READBACK=${EDGE_PROVISIONING_EDGE_READBACK:-}
KNOWN_HOSTS=${EDGE_PROVISIONING_KNOWN_HOSTS:-}
MACHINE_BASELINE=${EDGE_PROVISIONING_MACHINE_BASELINE:-}
APPROVED_PLAN_SHA256=${EDGE_PROVISIONING_APPROVED_PLAN_SHA256:-}
SEAL_SHA256=${EDGE_PROVISIONING_SEAL_SHA256:-}
READBACK_SHA256=${EDGE_PROVISIONING_EDGE_READBACK_SHA256:-}
KNOWN_HOST_FINGERPRINT=${EDGE_PROVISIONING_KNOWN_HOST_FINGERPRINT:-}
SNAPSHOT=${EDGE_PROVISIONING_SNAPSHOT:-}
SNAPSHOT_SHA256=${EDGE_PROVISIONING_SNAPSHOT_SHA256:-}
DELIVERY=${EDGE_PROVISIONING_DELIVERY:-}
DELIVERY_SHA256=${EDGE_PROVISIONING_DELIVERY_SHA256:-}

fail() { printf '%s\n' "$1" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: cloud-enrollment-smoke.sh --fixture --dry-run | --print-checklist | --delivery-gate [--snapshot PATH --delivery PATH] | --host happy-nursing-home-raw [--deploy --restart-check|--full-lifecycle --rollback-drill]' >&2
  exit 2
}
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
require_sha256() {
  value=$1
  label=$2
  case "$value" in *[!0-9a-f]*) fail "$label SHA-256 is invalid" ;; esac
  [ "${#value}" -eq 64 ] || fail "$label SHA-256 is invalid"
}
json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"));
for key in sys.argv[2:]: value=value[key]
assert isinstance(value, (str,int,bool)); print(str(value).lower() if isinstance(value,bool) else value)' "$@"
}
approved_sha() { awk -F': *' '$1 == "approved_plan_sha256" { print $2; exit }' "$1"; }
check_sealed_image() {
  repo=$1
  image=$2
  expected_repository=$3
  ref=$(json_value "$SEAL" "$repo" "$image" ref)
  image_id=$(json_value "$SEAL" "$repo" "$image" imageId)
  platform=$(json_value "$SEAL" "$repo" "$image" platform)
  revision=$(json_value "$SEAL" "$repo" "$image" revision)
  repository=$(json_value "$SEAL" "$repo" "$image" repository)
  case "$ref" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) fail "sealed $repo $image is not digest-pinned" ;; esac
  case "$image_id" in sha256:????????????????????????????????????????????????????????????????) ;; *) fail "sealed $repo $image ID is invalid" ;; esac
  case "$platform" in linux/amd64|linux/arm64) ;; *) fail "sealed $repo $image platform is invalid" ;; esac
  [ "$revision" = "$(json_value "$SEAL" "$repo" sha)" ] || fail "sealed $repo $image revision mismatch"
  [ "$repository" = "$expected_repository" ] || fail "sealed $repo $image repository mismatch"
}
gate_artifacts() {
  [ -f "$PLAN" ] && [ -f "$DRAFT" ] && [ -f "$SEAL" ] || fail 'approved plan, draft, and sealed RC receipt are required'
  require_sha256 "$APPROVED_PLAN_SHA256" 'approved plan anchor'
  require_sha256 "$SEAL_SHA256" 'sealed RC anchor'
  [ "$(sha256_file "$PLAN")" = "$APPROVED_PLAN_SHA256" ] || fail 'approved plan content changed'
  [ "$(approved_sha "$DRAFT")" = "$APPROVED_PLAN_SHA256" ] || fail 'draft approved-plan binding mismatch'
  grep -Fx 'round_status: approved' "$DRAFT" >/dev/null || fail 'draft review round is not approved'
  [ "$(sha256_file "$SEAL")" = "$SEAL_SHA256" ] || fail 'sealed RC content-address mismatch'
  [ "$(json_value "$SEAL" schemaVersion)" = 2 ] || fail 'sealed RC schema mismatch'
  [ "$(json_value "$SEAL" approvedPlanSha256)" = "$APPROVED_PLAN_SHA256" ] || fail 'sealed RC plan binding mismatch'
  [ "$(json_value "$SEAL" ai repository)" = SeniorAILab/eldercare-fall-ai ] || fail 'sealed AI repository identity mismatch'
  [ "$(json_value "$SEAL" ml repository)" = SeniorAILab/eldercare-fall-ml-v2 ] || fail 'sealed ML repository identity mismatch'
  for repo in ai ml; do
    sha=$(json_value "$SEAL" "$repo" sha)
    tree=$(json_value "$SEAL" "$repo" tree)
    case "$sha$tree" in *[!0-9a-f]*) fail "sealed $repo commit provenance is invalid" ;; esac
    [ "${#sha}" -eq 40 ] && [ "${#tree}" -eq 40 ] || fail "sealed $repo commit provenance is invalid"
  done
  check_sealed_image ai backendImage SeniorAILab/eldercare-fall-ai
  check_sealed_image ai frontImage SeniorAILab/eldercare-fall-ai
  check_sealed_image ml apiImage SeniorAILab/eldercare-fall-ml-v2
  check_sealed_image ml workerImage SeniorAILab/eldercare-fall-ml-v2
}
host_keys() {
  ssh-keygen -F "$HOST" -f "$KNOWN_HOSTS" 2>/dev/null | awk '$1 !~ /^#/ { print }'
}
host_fingerprint() {
  host_keys \
    | ssh-keygen -lf - -E sha256 2>/dev/null \
    | awk '{print $2}'
}
check_host_inputs() {
  [ "$HOST" = happy-nursing-home-raw ] || fail 'only happy-nursing-home-raw is approved; JNU targets are forbidden'
  [ -f "$KNOWN_HOSTS" ] || fail 'pinned known_hosts file is required'
  [ "$(host_keys | wc -l | tr -d ' ')" -eq 1 ] || fail 'known_hosts must contain exactly one approved host key'
  case "$KNOWN_HOST_FINGERPRINT" in SHA256:*) ;; *) fail 'independent host-key fingerprint is required' ;; esac
  [ "$(host_fingerprint)" = "$KNOWN_HOST_FINGERPRINT" ] || fail 'known_hosts fingerprint does not match independent anchor'
  [ -f "$MACHINE_BASELINE" ] || fail 'machine-id hash baseline is required'
  require_sha256 "$(sed -n '1p' "$MACHINE_BASELINE")" 'machine baseline'
}
check_snapshot() {
  section=$1
  [ "$(json_value "$READBACK" "$section" hostAlias)" = happy-nursing-home-raw ] || fail "$section host alias mismatch"
  case "$(json_value "$READBACK" "$section" hostname)" in *[Jj][Nn][Uu]*) fail 'JNU hostname is forbidden' ;; happy-*) ;; *) fail "$section hostname signature mismatch" ;; esac
  [ "$(json_value "$READBACK" "$section" machineIdSha256)" = "$(sed -n '1p' "$MACHINE_BASELINE")" ] || fail "$section machine-id hash mismatch"
  [ "$(json_value "$READBACK" "$section" deployLockAvailable)" = true ] || fail "$section deploy lock is held"
  [ "$(json_value "$READBACK" "$section" updaterIdle)" = true ] || fail "$section updater is concurrent"
  [ "$(json_value "$READBACK" "$section" clipStoreFreeMiB)" -ge "$MIN_EDGE_CLIP_MIB" ] || fail "$section clip-store free capacity is below 20 GiB"
  [ "$(json_value "$READBACK" "$section" volumesHealthy)" = true ] || fail "$section volumes are unhealthy"
  [ "$(json_value "$READBACK" "$section" queuesDrained)" = true ] || fail "$section queues are not drained"
  [ "$(json_value "$READBACK" "$section" sqliteBackupVerified)" = true ] || fail "$section SQLite backup is unverifiable"
  [ "$(json_value "$READBACK" "$section" apiImage)" = "$(json_value "$SEAL" ml apiImage ref)" ] || fail "$section API digest mismatch"
  [ "$(json_value "$READBACK" "$section" workerImage)" = "$(json_value "$SEAL" ml workerImage ref)" ] || fail "$section worker digest mismatch"
  [ "$(json_value "$READBACK" "$section" envFacilityIdentityAbsent)" = true ] || fail "$section facility identity remains in deployment env"
  [ "$(json_value "$READBACK" "$section" apiReady)" = true ] || fail "$section versioned API is unhealthy"
  [ "$(json_value "$READBACK" "$section" statusSchemaValid)" = true ] || fail "$section status schema is invalid"
  [ "$(json_value "$READBACK" "$section" schemaIntegrity)" = true ] || fail "$section SQLite schema integrity failed"
  [ "$(json_value "$READBACK" "$section" scopeVerified)" = true ] || fail "$section scope verification failed"
}
execution_ok() {
  execution_id=$1
  python3 - "$READBACK" "$execution_id" <<'PY'
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
receipt = Path(sys.argv[1])
body = json.loads(receipt.read_text(encoding="utf-8"))
item = next((value for value in body["executions"] if value.get("id") == sys.argv[2]), None)
assert item is not None and item.get("exitCode") == 0
assert isinstance(item.get("sequence"), int) and item["sequence"] > 0
assert re.fullmatch(r"[a-f0-9]{64}", item.get("evidenceSha256", ""))
assert item.get("evidencePath") == f"executions/{sys.argv[2]}.txt"
evidence = receipt.parent / item["evidencePath"]
assert hashlib.sha256(evidence.read_bytes()).hexdigest() == item["evidenceSha256"]
started = dt.datetime.fromisoformat(item["startedAt"].replace("Z", "+00:00"))
completed = dt.datetime.fromisoformat(item["completedAt"].replace("Z", "+00:00"))
assert completed >= started
PY
}
check_readback() {
  [ -f "$READBACK" ] || fail 'edge execution receipt is missing'
  require_sha256 "$READBACK_SHA256" 'edge execution receipt anchor'
  [ "$(sha256_file "$READBACK")" = "$READBACK_SHA256" ] || fail 'edge execution receipt content-address mismatch'
  [ "$(json_value "$READBACK" schemaVersion)" = 2 ] || fail 'edge execution receipt schema mismatch'
  check_snapshot preflight
  if [ "$FULL_LIFECYCLE" = true ]; then
    for id in enrollment topology-sync heartbeat event-clip credential-rotation timeout-retry rollback roll-forward restart; do execution_ok "$id" || fail "edge execution evidence failed: $id"; done
    check_snapshot postRestart
    python3 - "$READBACK" <<'PY'
import datetime as dt
import json
import sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
restart = next(value for value in body["executions"] if value["id"] == "restart")
observed = dt.datetime.fromisoformat(body["postRestart"]["observedAt"].replace("Z", "+00:00"))
completed = dt.datetime.fromisoformat(restart["completedAt"].replace("Z", "+00:00"))
assert body["postRestart"]["generation"] > body["preflight"]["generation"]
assert observed > completed
PY
  fi
}
delivery_fail() {
  printf 'CUTOVER_DELIVERY_FAIL reason=%s\n' "$1"
  exit 1
}
print_checklist() {
  printf '%s\n' 'CUTOVER_OPERATOR_CHECKLIST'
  printf '%s\n' \
    'item=enrollment' \
    'item=sealed-pre-cutover-snapshot' \
    'item=snapshot-derived-camera-count' \
    'item=non-empty-backend-mapping' \
    'item=mapping-pending-false' \
    'item=external-heartbeat-fresh' \
    'item=authenticated-sse-before-witness' \
    'item=no-fabricated-event' \
    'item=authenticated-clip' \
    'item=authenticated-vercel-read-side' \
    'item=edge-render-gid-matches-renderD128-owner' \
    'item=no-repository-gid-default' \
    'item=no-socket-privileged-docker-group-bypass' \
    'item=legacy-rollback-boundary-preserved'
  printf '%s\n' 'gid_contract=EDGE_RENDER_GID must equal the live /dev/dri/renderD128 owner GID; EDGE_VIDEO_GID is the live host video GID; there is no repository default'
}
check_delivery_gate() {
  [ -n "$SNAPSHOT" ] || delivery_fail missing-snapshot
  [ -n "$DELIVERY" ] || delivery_fail missing-delivery
  [ -f "$SNAPSHOT" ] || delivery_fail missing-snapshot
  [ -f "$DELIVERY" ] || delivery_fail missing-delivery
  case "$SNAPSHOT_SHA256" in *[!0-9a-f]*) delivery_fail malformed-receipt ;; esac
  [ "${#SNAPSHOT_SHA256}" -eq 64 ] || delivery_fail malformed-receipt
  case "$DELIVERY_SHA256" in *[!0-9a-f]*) delivery_fail malformed-receipt ;; esac
  [ "${#DELIVERY_SHA256}" -eq 64 ] || delivery_fail malformed-receipt
  [ "$(sha256_file "$SNAPSHOT")" = "$SNAPSHOT_SHA256" ] || delivery_fail dirty-snapshot
  [ "$(sha256_file "$DELIVERY")" = "$DELIVERY_SHA256" ] || delivery_fail dirty-delivery
  python3 - "$SNAPSHOT" "$DELIVERY" <<'PY'
import datetime as dt
import json
import re
import sys

snapshot_path, delivery_path = sys.argv[1], sys.argv[2]

class GateError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

LOCAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
BACKEND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
EVENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
SECRET_KEYS = {
    "password",
    "token",
    "secret",
    "authorization",
    "rtsp",
    "rtsps",
    "credential",
    "api_key",
    "apikey",
}
SECRET_VALUE = re.compile(r"(rtsp://|rtsps://|password=|token=|bearer\s)", re.I)

def fail(reason: str) -> None:
    raise GateError(reason)

def load(path: str) -> object:
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        fail("malformed-receipt")

def walk(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in SECRET_KEYS or any(part in lowered for part in ("password", "token", "secret", "rtsp")):
                fail("secret-bearing")
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, str) and SECRET_VALUE.search(value):
        fail("secret-bearing")

def parse_time(raw: object, reason: str) -> dt.datetime:
    if not isinstance(raw, str):
        fail(reason)
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        fail(reason)

def require_opaque(value: object, reason: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        fail(reason)
    if value != value.strip() or not pattern.fullmatch(value):
        fail(reason)
    lowered = value.lower()
    if "room" in lowered or " " in value or "/" in value:
        fail(reason)
    return value

try:
    snapshot = load(snapshot_path)
    delivery = load(delivery_path)
    walk(snapshot)
    walk(delivery)
    if not isinstance(snapshot, dict) or not isinstance(delivery, dict):
        fail("malformed-receipt")
    if snapshot.get("schemaVersion") != 1 or snapshot.get("kind") != "pre-cutover-snapshot":
        fail("malformed-receipt")
    if delivery.get("schemaVersion") != 1 or delivery.get("kind") != "cutover-delivery-readout":
        fail("malformed-receipt")
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        fail("malformed-receipt")
    expected = snapshot.get("cameraCount")
    if not isinstance(expected, int) or expected != len(cameras) or expected < 1:
        fail("count-drift")
    expected_ids: list[str] = []
    seen: set[str] = set()
    for row in cameras:
        if not isinstance(row, dict):
            fail("malformed-receipt")
        local_id = require_opaque(row.get("localId"), "malformed-id", LOCAL_RE)
        if local_id in seen:
            fail("malformed-id")
        seen.add(local_id)
        expected_ids.append(local_id)
        klass = row.get("class")
        if klass not in {"expected", "expected-witness"}:
            fail("malformed-receipt")
    witness = require_opaque(snapshot.get("designatedWitnessLocalId"), "malformed-id", LOCAL_RE)
    if witness not in seen:
        fail("malformed-id")
    enrollment = delivery.get("enrollment")
    if not isinstance(enrollment, dict):
        fail("malformed-receipt")
    if enrollment.get("ok") is not True or enrollment.get("authenticated") is not True:
        fail("auth-absent")
    observed = delivery.get("cameras")
    if not isinstance(observed, list):
        fail("malformed-receipt")
    if len(observed) != expected:
        fail("count-drift")
    mapped = 0
    heartbeats = 0
    seen_delivery: set[str] = set()
    seen_backends: set[str] = set()
    for row in observed:
        if not isinstance(row, dict):
            fail("malformed-receipt")
        local_id = require_opaque(row.get("localId"), "malformed-id", LOCAL_RE)
        if local_id not in seen or local_id in seen_delivery:
            fail("count-drift")
        seen_delivery.add(local_id)
        backend = row.get("backendCameraId")
        if not isinstance(backend, str) or not backend.strip():
            fail("unmapped-camera" if backend in (None, "") else "blank-mapping")
        if backend != backend.strip():
            fail("blank-mapping")
        require_opaque(backend, "malformed-id", BACKEND_RE)
        if backend in seen_backends:
            fail("duplicate-mapping")
        seen_backends.add(backend)
        if row.get("mappingPending") is True:
            fail("mapping-pending")
        if row.get("mappingPending") is not False:
            fail("malformed-receipt")
        heartbeat = row.get("externalHeartbeat")
        if not isinstance(heartbeat, dict):
            fail("missed-heartbeat")
        if heartbeat.get("ok") is not True:
            fail("missed-heartbeat")
        if heartbeat.get("fresh") is not True:
            fail("stale-heartbeat")
        mapped += 1
        heartbeats += 1
    if seen_delivery != seen:
        fail("count-drift")
    sse = delivery.get("sse")
    if not isinstance(sse, dict):
        fail("malformed-receipt")
    if sse.get("authenticated") is not True:
        fail("auth-absent")
    sse_at = parse_time(sse.get("establishedAt"), "sse-order")
    witness_body = delivery.get("witness")
    if not isinstance(witness_body, dict):
        fail("malformed-receipt")
    observed_witness = require_opaque(witness_body.get("localId"), "malformed-id", LOCAL_RE)
    if observed_witness != witness:
        fail("malformed-id")
    processing_at = parse_time(witness_body.get("processingEnabledAt"), "sse-order")
    if sse.get("establishedBeforeWitnessProcessing") is not True or sse_at >= processing_at:
        fail("sse-order")
    if witness_body.get("eventFabricated") is True or witness_body.get("realEvent") is not True:
        fail("fabricated-event")
    require_opaque(witness_body.get("edgeEventId"), "malformed-id", EVENT_RE)
    if witness_body.get("clipAuthenticated") is not True:
        fail("clip-unauthenticated")
    if witness_body.get("vercelReadSideAuthenticated") is not True:
        fail("vercel-unauthenticated")
    print(f"CUTOVER_DELIVERY_OK expected={expected} mapped={mapped} heartbeats={heartbeats}")
except GateError as exc:
    print(f"CUTOVER_DELIVERY_FAIL reason={exc.reason}")
    raise SystemExit(1)
PY
}
write_delivery_pair() {
  dest=$1
  expected=$2
  mapped=$3
  heartbeat_ok=$4
  share_backend=${5:-0}
  python3 - "$dest" "$expected" "$mapped" "$heartbeat_ok" "$share_backend" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
expected = int(sys.argv[2])
mapped = int(sys.argv[3])
heartbeat_ok = sys.argv[4] == "1"
share_backend = sys.argv[5] == "1"
ids = [
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
][:expected]
backends = [
    "be-aaaa1111bbbb2222cccc3333dddd4444",
    "be-eeee5555ffff6666aaaa7777bbbb8888",
    "be-9999cccc0000dddd1111eeee2222ffff",
]
if share_backend and expected >= 2:
    backends[1] = backends[0]
witness = ids[-1]
snapshot = {
    "schemaVersion": 1,
    "kind": "pre-cutover-snapshot",
    "cameraCount": expected,
    "cameras": [
        {
            "localId": camera_id,
            "class": "expected-witness" if camera_id == witness else "expected",
        }
        for camera_id in ids
    ],
    "designatedWitnessLocalId": witness,
}
cameras = []
for index, camera_id in enumerate(ids):
    backend = backends[index] if index < mapped else ""
    cameras.append(
        {
            "localId": camera_id,
            "backendCameraId": backend,
            "mappingPending": False,
            "externalHeartbeat": {
                "ok": heartbeat_ok if index < mapped else False,
                "fresh": True,
            },
        }
    )
delivery = {
    "schemaVersion": 1,
    "kind": "cutover-delivery-readout",
    "enrollment": {"ok": True, "authenticated": True},
    "cameras": cameras,
    "sse": {
        "authenticated": True,
        "establishedAt": "2026-08-11T00:00:00Z",
        "establishedBeforeWitnessProcessing": True,
    },
    "witness": {
        "localId": witness,
        "processingEnabledAt": "2026-08-11T00:00:02Z",
        "eventFabricated": False,
        "realEvent": True,
        "edgeEventId": "evt-aaaabbbbccccddddeeeeffff00001111",
        "clipAuthenticated": True,
        "vercelReadSideAuthenticated": True,
    },
}
dest.mkdir(parents=True, exist_ok=True)
(dest / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(dest / "delivery.json").write_text(json.dumps(delivery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
run_delivery_case() {
  dest=$1
  SNAPSHOT=$dest/snapshot.json
  DELIVERY=$dest/delivery.json
  SNAPSHOT_SHA256=$(sha256_file "$SNAPSHOT")
  DELIVERY_SHA256=$(sha256_file "$DELIVERY")
  check_delivery_gate
}
fixture() {
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/cloud-enrollment-smoke.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT HUP INT TERM
  printf 'fixture approved plan\n' >"$tmp/plan.md"
  APPROVED_PLAN_SHA256=$(sha256_file "$tmp/plan.md")
  printf 'approved_plan_sha256: %s\nround_status: approved\n' "$APPROVED_PLAN_SHA256" >"$tmp/draft.md"
  cat >"$tmp/seal.json" <<EOF
{"schemaVersion":2,"approvedPlanSha256":"$APPROVED_PLAN_SHA256","ai":{"repository":"SeniorAILab/eldercare-fall-ai","sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tree":"1111111111111111111111111111111111111111","backendImage":{"ref":"local/backend@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","imageId":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","platform":"linux/arm64","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repository":"SeniorAILab/eldercare-fall-ai"},"frontImage":{"ref":"local/front@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","imageId":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","platform":"linux/arm64","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repository":"SeniorAILab/eldercare-fall-ai"}},"ml":{"repository":"SeniorAILab/eldercare-fall-ml-v2","sha":"dddddddddddddddddddddddddddddddddddddddd","tree":"2222222222222222222222222222222222222222","apiImage":{"ref":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","imageId":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","platform":"linux/arm64","revision":"dddddddddddddddddddddddddddddddddddddddd","repository":"SeniorAILab/eldercare-fall-ml-v2"},"workerImage":{"ref":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","imageId":"sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","platform":"linux/amd64","revision":"dddddddddddddddddddddddddddddddddddddddd","repository":"SeniorAILab/eldercare-fall-ml-v2"}}}
EOF
  printf fixture-machine | sha256_file /dev/stdin >"$tmp/machine.sha256"
  ssh-keygen -q -t ed25519 -N '' -f "$tmp/host-key"
  printf 'happy-nursing-home-raw %s\n' "$(cat "$tmp/host-key.pub")" >"$tmp/known_hosts"
  KNOWN_HOST_FINGERPRINT=$(ssh-keygen -lf "$tmp/host-key.pub" -E sha256 | awk '{print $2}')
  machine=$(cat "$tmp/machine.sha256")
  snapshot='"hostAlias":"happy-nursing-home-raw","hostname":"happy-edge-fixture","machineIdSha256":"'"$machine"'","deployLockAvailable":true,"updaterIdle":true,"clipStoreFreeMiB":20480,"volumesHealthy":true,"queuesDrained":true,"sqliteBackupVerified":true,"apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","envFacilityIdentityAbsent":true,"apiReady":true,"statusSchemaValid":true,"schemaIntegrity":true,"scopeVerified":true'
  executions=
  sequence=0
  mkdir "$tmp/executions" "$tmp/state"
  for id in enrollment topology-sync heartbeat event-clip credential-rotation timeout-retry rollback roll-forward restart; do
    sequence=$((sequence + 1))
    case "$id" in
      rollback) printf 'before\n' >"$tmp/state/runtime"; cp "$tmp/state/runtime" "$tmp/state/snapshot"; printf 'changed\n' >"$tmp/state/runtime"; cp "$tmp/state/snapshot" "$tmp/state/runtime"; cmp "$tmp/state/runtime" "$tmp/state/snapshot" ;;
      roll-forward) printf 'forward\n' >"$tmp/state/runtime" ;;
      restart) printf '2\n' >"$tmp/state/generation" ;;
      *) printf '%s\n' "$id" >"$tmp/state/$id" ;;
    esac >"$tmp/executions/$id.txt" 2>&1
    printf 'action=%s exit=0\n' "$id" >>"$tmp/executions/$id.txt"
    evidence_sha=$(sha256_file "$tmp/executions/$id.txt")
    row="{\"id\":\"$id\",\"sequence\":$sequence,\"startedAt\":\"2026-08-11T00:00:00Z\",\"completedAt\":\"2026-08-11T00:00:01Z\",\"exitCode\":0,\"evidencePath\":\"executions/$id.txt\",\"evidenceSha256\":\"$evidence_sha\"}"
    [ -z "$executions" ] && executions=$row || executions="$executions,$row"
  done
  printf '{"schemaVersion":2,"preflight":{"generation":1,"observedAt":"2026-08-10T23:59:59Z",%s},"executions":[%s],"postRestart":{"generation":2,"observedAt":"2026-08-11T00:00:02Z",%s}}\n' "$snapshot" "$executions" "$snapshot" >"$tmp/readback.json"
  PLAN=$tmp/plan.md DRAFT=$tmp/draft.md SEAL=$tmp/seal.json READBACK=$tmp/readback.json KNOWN_HOSTS=$tmp/known_hosts MACHINE_BASELINE=$tmp/machine.sha256 HOST=happy-nursing-home-raw
  SEAL_SHA256=$(sha256_file "$SEAL") READBACK_SHA256=$(sha256_file "$READBACK") FULL_LIFECYCLE=true
  gate_artifacts
  check_host_inputs
  check_readback
  cp "$tmp/executions/restart.txt" "$tmp/restart.txt.original"
  printf 'forged\n' >>"$tmp/executions/restart.txt"
  if (check_readback) >/dev/null 2>&1; then fail 'forged execution evidence passed'; fi
  mv "$tmp/restart.txt.original" "$tmp/executions/restart.txt"
  printf '%s\n' 'EDGE_EXECUTION_CONTENT_ADDRESS_REJECTION_OK'
  cp "$SEAL" "$tmp/bad-seal.json"
  python3 - "$tmp/bad-seal.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["ml"]["workerImage"]["repository"] = "attacker/repository"
path.write_text(json.dumps(value), encoding="utf-8")
PY
  original_seal=$SEAL original_seal_sha=$SEAL_SHA256 SEAL=$tmp/bad-seal.json SEAL_SHA256=$(sha256_file "$tmp/bad-seal.json")
  if (gate_artifacts) >/dev/null 2>&1; then fail 'invalid image provenance passed'; fi
  SEAL=$original_seal SEAL_SHA256=$original_seal_sha
  printf '%s\n' 'EDGE_IMAGE_PROVENANCE_REJECTION_OK'
  cp "$KNOWN_HOSTS" "$tmp/known_hosts.original"
  ssh-keygen -q -t ed25519 -N '' -f "$tmp/extra-host-key"
  printf 'happy-nursing-home-raw %s\n' "$(cat "$tmp/extra-host-key.pub")" >>"$KNOWN_HOSTS"
  if (check_host_inputs) >/dev/null 2>&1; then fail 'duplicate accepted host key passed'; fi
  mv "$tmp/known_hosts.original" "$KNOWN_HOSTS"
  printf '%s\n' 'EDGE_DUPLICATE_HOST_KEY_REJECTION_OK'
  cp "$READBACK" "$tmp/tampered.json"
  python3 - "$tmp/tampered.json" <<'PY'
import json, sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["postRestart"]["queuesDrained"] = False
json.dump(value, open(path, "w", encoding="utf-8"))
PY
  READBACK=$tmp/tampered.json
  if (check_readback) >/dev/null 2>&1; then fail 'tampered post-restart receipt passed'; fi
  READBACK_SHA256=$(sha256_file "$READBACK")
  if (check_readback) >/dev/null 2>&1; then fail 'stale post-restart state passed'; fi
  KNOWN_HOST_FINGERPRINT=SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
  if (check_host_inputs) >/dev/null 2>&1; then fail 'spoofed known_hosts fingerprint passed'; fi
  write_delivery_pair "$tmp/delivery-all" 3 3 1
  run_delivery_case "$tmp/delivery-all" >/dev/null
  printf '%s\n' 'EDGE_DELIVERY_ALL_MAPPED_OK'
  write_delivery_pair "$tmp/delivery-unmapped" 3 2 1
  if (run_delivery_case "$tmp/delivery-unmapped") >/dev/null 2>&1; then fail 'unmapped camera passed'; fi
  printf '%s\n' 'EDGE_DELIVERY_UNMAPPED_REJECTION_OK'
  write_delivery_pair "$tmp/delivery-missed" 3 3 0
  if (run_delivery_case "$tmp/delivery-missed") >/dev/null 2>&1; then fail 'missed heartbeat passed'; fi
  printf '%s\n' 'EDGE_DELIVERY_MISSED_HEARTBEAT_REJECTION_OK'
  write_delivery_pair "$tmp/delivery-duplicate" 2 2 1 1
  if (run_delivery_case "$tmp/delivery-duplicate") >/dev/null 2>&1; then fail 'duplicate backend mapping passed'; fi
  printf '%s\n' 'EDGE_DELIVERY_DUPLICATE_MAPPING_REJECTION_OK'
  if (sh "$0" --print-checklist --delivery-gate) >/dev/null 2>&1; then fail 'combined checklist and delivery-gate passed'; fi
  printf '%s\n' 'EDGE_CHECKLIST_DELIVERY_EXCLUSIVE_OK'
  checklist=$(print_checklist)
  printf '%s\n' "$checklist" | grep -F 'item=edge-render-gid-matches-renderD128-owner' >/dev/null
  printf '%s\n' "$checklist" | grep -F 'EDGE_RENDER_GID' >/dev/null
  printf '%s\n' "$checklist" | grep -F 'renderD128' >/dev/null
  printf '%s\n' "$checklist" | grep -E '(^|[^0-9])104([^0-9]|$)' >/dev/null && fail 'checklist still cites 104' || true
  printf '%s\n' "$checklist" | grep -E '(^|[^0-9])44([^0-9]|$)' >/dev/null && fail 'checklist still cites 44' || true
  printf '%s\n' 'CUTOVER_CHECKLIST_GID_CONTRACT_OK'
  printf '%s\n' 'CLOUD_ENROLLMENT_SMOKE_FIXTURE_OK'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fixture) MODE=fixture; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --print-checklist) PRINT_CHECKLIST=true; shift ;;
    --delivery-gate) DELIVERY_GATE=true; shift ;;
    --snapshot) [ "$#" -ge 2 ] || usage; SNAPSHOT=$2; shift 2 ;;
    --delivery) [ "$#" -ge 2 ] || usage; DELIVERY=$2; shift 2 ;;
    --host) [ "$#" -ge 2 ] || usage; HOST=$2; shift 2 ;;
    --deploy) DEPLOY=true; shift ;;
    --restart-check) RESTART_CHECK=true; shift ;;
    --full-lifecycle) FULL_LIFECYCLE=true; shift ;;
    --rollback-drill) ROLLBACK_DRILL=true; shift ;;
    *) usage ;;
  esac
done
if [ "$PRINT_CHECKLIST" = true ] && [ "$DELIVERY_GATE" = true ]; then
  fail 'print-checklist and delivery-gate are mutually exclusive'
fi
[ "$PRINT_CHECKLIST" = true ] && { print_checklist; exit 0; }
[ "$MODE" = fixture ] && { [ "$DRY_RUN" = true ] || usage; fixture; exit 0; }
[ "$DELIVERY_GATE" = true ] && { check_delivery_gate; exit 0; }
gate_artifacts
check_host_inputs
check_readback
if [ "$DEPLOY" = true ]; then
  [ "$RESTART_CHECK" = true ] || usage
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o CheckHostIP=yes -o UserKnownHostsFile="$KNOWN_HOSTS" "$HOST" \
    'sudo -n /opt/eldercare-fall-ml/scripts/edge-updater/update-edge.sh'
  api_image=$(json_value "$SEAL" ml apiImage ref)
  worker_image=$(json_value "$SEAL" ml workerImage ref)
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o CheckHostIP=yes -o UserKnownHostsFile="$KNOWN_HOSTS" "$HOST" \
    sh -s -- "$api_image" "$worker_image" <<'REMOTE'
set -eu
api_image=$1
worker_image=$2
api_digest=${api_image##*@}
worker_digest=${worker_image##*@}
cd /opt/eldercare-fall-ml
sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml restart ml-api ml-worker >/dev/null
attempt=0
until curl -fsS http://127.0.0.1:8000/health/ready >/dev/null; do
  attempt=$((attempt + 1)); [ "$attempt" -lt 30 ] || exit 1; sleep 1
done
api_id=$(sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml ps -q ml-api)
worker_id=$(sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml ps -q ml-worker)
[ -n "$api_id" ] && [ -n "$worker_id" ]
[ "$(sudo -n docker inspect --format '{{.State.Running}}' "$api_id")" = true ]
[ "$(sudo -n docker inspect --format '{{.State.Running}}' "$worker_id")" = true ]
[ "$(sudo -n docker inspect --format '{{.Config.Image}}' "$api_id")" = "$api_image" ]
[ "$(sudo -n docker inspect --format '{{.Config.Image}}' "$worker_id")" = "$worker_image" ]
facility_identity_key=$(printf '%s%s' 'API_FACILITY_' 'ID')
facility_token_key=$(printf '%s%s' 'EDGE_FACILITY_' 'TOKEN')
sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml config \
  | grep -Eq "$facility_identity_key|$facility_token_key" && exit 1 || true
curl -fsS http://127.0.0.1:8000/api/v1/status | python3 -c 'import json,sys; value=json.load(sys.stdin); assert isinstance(value["cameras"],dict); assert isinstance(value["runtime"],dict)'
curl -fsS http://127.0.0.1:8000/api/v1/system | python3 -c '
import json, sys
api_digest, worker_digest = sys.argv[1:]
value = json.load(sys.stdin)
assert value["image_digests"]["ml_api"] == api_digest
assert value["image_digests"]["ml_worker"] == worker_digest
assert value["backend"]["configured"] is True
' "$api_digest" "$worker_digest"
REMOTE
  check_readback
fi
if [ "$FULL_LIFECYCLE" = true ]; then [ "$ROLLBACK_DRILL" = true ] || usage; fi
printf 'CLOUD_ENROLLMENT_SMOKE_OK ml_sha=%s mode=authenticated-execution\n' "$(json_value "$SEAL" ml sha)"
