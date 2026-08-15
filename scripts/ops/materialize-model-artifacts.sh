#!/bin/bash
# Pre-boot model materialization from an operator-private hash-only receipt.
#
# Verifies the approved source image digest/revision immediately before a
# normal Docker-CLI copy, copies exactly the declared relative paths, then
# checks destination SHA-256 values and sealed-checkout sidecars. Emits only
# a count/path-class/hash verdict. Never prints credentials, RTSP, model
# bytes, source identity, or destination paths.
set -euo pipefail
set +x

usage() {
  printf '%s\n' 'Usage: scripts/ops/materialize-model-artifacts.sh --receipt PATH --dest PATH --checkout PATH' >&2
  exit 2
}

fail() {
  printf 'MODEL_MATERIALIZATION_FAIL reason=%s\n' "$1"
  exit 1
}

RECEIPT=""
DEST=""
CHECKOUT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --receipt)
      [ "$#" -ge 2 ] || usage
      RECEIPT=$2
      shift 2
      ;;
    --dest)
      [ "$#" -ge 2 ] || usage
      DEST=$2
      shift 2
      ;;
    --checkout)
      [ "$#" -ge 2 ] || usage
      CHECKOUT=$2
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

[ -n "$RECEIPT" ] && [ -n "$DEST" ] && [ -n "$CHECKOUT" ] || usage
[ -f "$RECEIPT" ] || fail malformed-receipt
[ -d "$CHECKOUT" ] || fail missing-sidecar

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
VERIFY="$SCRIPT_DIR/verify-model-artifacts.sh"
DOCKER_BIN=${SEEON_MODEL_DOCKER:-docker}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/seeon-model-materialize.XXXXXX")
CREATED_FILES="$WORK/created.files"
: >"$CREATED_FILES"
COMMITTED=0
PUBLISH_SEQ=0
cleanup() {
  if [ "${COMMITTED:-0}" -eq 0 ] && [ -n "${DEST:-}" ]; then
    if [ -f "${CREATED_FILES:-}" ]; then
      while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        rm -f "$DEST/$rel" >/dev/null 2>&1 || true
      done <"$CREATED_FILES"
    fi
    if [ -d "$DEST" ] && [ ! -L "$DEST" ]; then
      find -P "$DEST" -maxdepth 1 -type f -name '.seeon-pub.*' -exec rm -f {} +
    fi
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'fail copy-failed' INT TERM HUP

set +e
python3 - "$RECEIPT" "$WORK" <<'PY'
import json
import re
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
sha_re = re.compile(r"^[0-9a-f]{64}$")
rev_re = re.compile(r"^[0-9a-f]{40}$")
name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
root_re = re.compile(r"^/[A-Za-z0-9/._-]+$")
rel_re = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]*$")
allowed_top = {"schemaVersion", "source", "artifacts", "sidecars"}
allowed_source = {"kind", "container", "imageDigest", "revision", "root"}
allowed_item = {"path", "sha256", "class"}


def die() -> None:
    raise SystemExit(1)


try:
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    die()
if not isinstance(body, dict) or set(body) != allowed_top:
    die()
if body.get("schemaVersion") != 1:
    die()
source = body.get("source")
if not isinstance(source, dict) or set(source) != allowed_source:
    die()
if source.get("kind") != "docker-cli":
    die()
container = source.get("container")
digest = source.get("imageDigest")
revision = source.get("revision")
root = source.get("root")
if not isinstance(container, str) or not name_re.fullmatch(container):
    die()
if not isinstance(digest, str) or not sha_re.fullmatch(digest):
    die()
if not isinstance(revision, str) or not rev_re.fullmatch(revision):
    die()
if not isinstance(root, str) or not root_re.fullmatch(root) or ".." in root.split("/"):
    die()
artifacts = body.get("artifacts")
sidecars = body.get("sidecars")
if not isinstance(artifacts, list) or not artifacts or not isinstance(sidecars, list):
    die()

seen: set[str] = set()
rows: list[tuple[str, str, str]] = []
for items, allowed_class in ((artifacts, {"weight", "provenance"}), (sidecars, {"sidecar"})):
    for item in items:
        if not isinstance(item, dict) or set(item) != allowed_item:
            die()
        path = item.get("path")
        digest_value = item.get("sha256")
        klass = item.get("class")
        if not isinstance(path, str) or not isinstance(digest_value, str) or not isinstance(klass, str):
            die()
        if not rel_re.fullmatch(path) or "//" in path or path.endswith("/"):
            die()
        if any(part in {".", ".."} for part in path.split("/")):
            raise SystemExit(3)
        if not sha_re.fullmatch(digest_value) or klass not in allowed_class:
            die()
        if path in seen:
            die()
        seen.add(path)
        rows.append((path, digest_value, klass))

(out_dir / "source.env").write_text(
    f"CONTAINER={container}\nIMAGE_DIGEST={digest}\nREVISION={revision}\nSOURCE_ROOT={root}\n",
    encoding="utf-8",
)
artifact_lines = []
sidecar_lines = []
for path, digest_value, klass in rows:
    line = f"{path}\t{digest_value}\t{klass}\n"
    if klass == "sidecar":
        sidecar_lines.append(line)
    else:
        artifact_lines.append(line)
(out_dir / "artifacts.tsv").write_text("".join(artifact_lines), encoding="utf-8")
(out_dir / "sidecars.tsv").write_text("".join(sidecar_lines), encoding="utf-8")
raise SystemExit(0)
PY
status=$?
set -e
if [ "$status" -eq 3 ]; then
  fail path-traversal
elif [ "$status" -ne 0 ]; then
  fail malformed-receipt
fi

# shellcheck disable=SC1091
. "$WORK/source.env"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if ! git -C "$CHECKOUT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  fail dirty-checkout
fi
if [ -n "$(git -C "$CHECKOUT" status --porcelain)" ]; then
  fail dirty-checkout
fi

assert_dest_component_safe() {
  local root=$1 rel=$2 current rest part
  if [ -L "$root" ]; then
    fail dest-symlink
  fi
  if [ -e "$root" ] && [ ! -d "$root" ]; then
    fail dest-not-directory
  fi
  current=$root
  rest=$rel
  while [ -n "$rest" ]; do
    case "$rest" in
      */*)
        part=${rest%%/*}
        rest=${rest#*/}
        ;;
      *)
        part=$rest
        rest=""
        ;;
    esac
    [ -n "$part" ] || fail path-traversal
    current="$current/$part"
    if [ -L "$current" ]; then
      fail dest-symlink
    fi
    if [ -n "$rest" ]; then
      if [ -e "$current" ] && [ ! -d "$current" ]; then
        fail dest-not-directory
      fi
    elif [ -e "$current" ] && [ ! -f "$current" ]; then
      fail dest-not-directory
    fi
  done
}

assert_destination_safe() {
  if [ -L "$DEST" ]; then
    fail dest-symlink
  fi
  if [ -e "$DEST" ] && [ ! -d "$DEST" ]; then
    fail dest-not-directory
  fi
  if [ ! -e "$DEST" ]; then
    return 0
  fi
  while IFS="$(printf '\t')" read -r rel _expected _klass; do
    [ -n "$rel" ] || continue
    assert_dest_component_safe "$DEST" "$rel"
  done < <(cat "$WORK/artifacts.tsv" "$WORK/sidecars.tsv")
}

assert_destination_safe

STAGE="$WORK/stage"
mkdir -p "$STAGE" >/dev/null 2>&1 || fail copy-failed
dest_map="$WORK/dest.map"
: >"$dest_map"

while IFS="$(printf '\t')" read -r rel expected _klass; do
  [ -n "$rel" ] || continue
  if ! git -C "$CHECKOUT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
    fail missing-sidecar
  fi
  src="$CHECKOUT/$rel"
  if [ -L "$src" ]; then
    fail sidecar-symlink
  fi
  if [ ! -f "$src" ]; then
    fail missing-sidecar
  fi
  actual=$(sha256_file "$src")
  if [ "$actual" != "$expected" ]; then
    fail missing-sidecar
  fi
  mkdir -p "$STAGE/$(dirname -- "$rel")" >/dev/null 2>&1 || fail copy-failed
  cp "$src" "$STAGE/$rel" >/dev/null 2>&1 || fail copy-failed
  printf '%s\t%s\n' "$rel" "$actual" >>"$dest_map"
done <"$WORK/sidecars.tsv"

inspect_out=""
inspect_status=0
inspect_out=$("$DOCKER_BIN" inspect --format '{{.Image}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "$CONTAINER" 2>/dev/null) || inspect_status=$?
inspect_out=$(printf '%s' "$inspect_out" | tr -d '\r' | awk 'NR==1 {print; exit}')
expected_inspect="sha256:${IMAGE_DIGEST} ${REVISION}"
if [ "$inspect_status" -ne 0 ]; then
  fail source-identity
fi
if [ "$inspect_out" != "$expected_inspect" ]; then
  case "$inspect_out" in
    sha256:[0-9a-f][0-9a-f]*)
      fail wrong-image
      ;;
    *)
      fail source-identity
      ;;
  esac
fi

while IFS="$(printf '\t')" read -r rel expected _klass; do
  [ -n "$rel" ] || continue
  mkdir -p "$STAGE/$(dirname -- "$rel")" >/dev/null 2>&1 || fail copy-failed
  copy_status=0
  copy_err=$("$DOCKER_BIN" cp "$CONTAINER:$SOURCE_ROOT/$rel" "$STAGE/$rel" 2>&1) || copy_status=$?
  copy_err=$(printf '%s' "$copy_err" | tr -d '\r')
  if [ "$copy_status" -ne 0 ]; then
    case "$copy_err" in
      *"No such file"*)
        fail missing-artifact
        ;;
      *)
        fail copy-failed
        ;;
    esac
  fi
  if [ ! -f "$STAGE/$rel" ]; then
    fail missing-artifact
  fi
  actual=$(sha256_file "$STAGE/$rel")
  if [ "$actual" != "$expected" ]; then
    fail altered-artifact
  fi
  printf '%s\t%s\n' "$rel" "$actual" >>"$dest_map"
done <"$WORK/artifacts.tsv"

assert_destination_safe
if [ ! -e "$DEST" ]; then
  mkdir -p "$DEST" >/dev/null 2>&1 || fail dest-not-directory
fi
if [ -L "$DEST" ] || [ ! -d "$DEST" ]; then
  if [ -L "$DEST" ]; then
    fail dest-symlink
  fi
  fail dest-not-directory
fi
while IFS="$(printf '\t')" read -r rel expected _klass; do
  [ -n "$rel" ] || continue
  mkdir -p "$DEST/$(dirname -- "$rel")" >/dev/null 2>&1 || fail dest-not-directory
  if [ -L "$DEST/$(dirname -- "$rel")" ] || [ -L "$DEST/$rel" ]; then
    fail dest-symlink
  fi
  PUBLISH_SEQ=$((PUBLISH_SEQ + 1))
  tmp_rel=".seeon-pub.${PUBLISH_SEQ}"
  tmp_path="$DEST/$tmp_rel"
  printf '%s\n' "$tmp_rel" >>"$CREATED_FILES"
  if [ "${SEEON_MODEL_TEST_PUBLISH_SIGNAL:-}" = "TERM" ]; then
    kill -s TERM "$$"
    fail copy-failed
  fi
  if [ "${SEEON_MODEL_TEST_PUBLISH_FAIL:-}" = "1" ]; then
    fail copy-failed
  fi
  cp "$STAGE/$rel" "$tmp_path" >/dev/null 2>&1 || fail copy-failed
  if [ -L "$tmp_path" ] || [ ! -f "$tmp_path" ]; then
    fail copy-failed
  fi
  actual=$(sha256_file "$tmp_path")
  if [ "$actual" != "$expected" ]; then
    fail altered-artifact
  fi
  mv -f "$tmp_path" "$DEST/$rel" >/dev/null 2>&1 || fail copy-failed
  printf '%s\n' "$rel" >>"$CREATED_FILES"
done < <(cat "$WORK/artifacts.tsv" "$WORK/sidecars.tsv")

count=$(grep -c . "$WORK/artifacts.tsv" || true)
sidecar_count=$(grep -c . "$WORK/sidecars.tsv" || true)
dest_sha256=$(python3 - "$dest_map" <<'PY'
import hashlib
import sys
from pathlib import Path

mapping = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    path, digest = line.split("\t", 1)
    mapping[path] = digest
payload = "".join(f"{path}\t{digest}\n" for path, digest in sorted(mapping.items()))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
PY
)

verify_out=$("$VERIFY" --receipt "$RECEIPT" --dest "$DEST" --checkout "$CHECKOUT") || fail copy-failed
printf '%s\n' "$verify_out" | grep -q "^MODEL_VERIFY_OK " || fail copy-failed
COMMITTED=1

printf 'MODEL_MATERIALIZATION_OK count=%s sidecar_count=%s dest_sha256=%s\n' \
  "$count" "$sidecar_count" "$dest_sha256"
