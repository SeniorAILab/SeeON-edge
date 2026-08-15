#!/bin/bash
# Verify a destination model tree against an operator-private hash-only receipt.
#
# Pre-boot only. Emits a count/sidecar_count/dest_sha256 verdict. Never prints
# source identity, destination paths, credentials, or artifact bytes.
set -euo pipefail
set +x

usage() {
  printf '%s\n' 'Usage: scripts/ops/verify-model-artifacts.sh --receipt PATH --dest PATH --checkout PATH' >&2
  exit 2
}

fail() {
  printf 'MODEL_VERIFY_FAIL reason=%s\n' "$1"
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
if [ -L "$DEST" ]; then
  fail dest-symlink
fi
[ -d "$DEST" ] || fail missing-artifact
[ -d "$CHECKOUT" ] || fail missing-sidecar

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
PARSE="$SCRIPT_DIR/parse-model-receipt.py"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/seeon-model-verify.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

set +e
python3 "$PARSE" "$RECEIPT" "$WORK"
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

if find -P "$DEST" -type l -print 2>/dev/null | grep -q .; then
  fail dest-symlink
fi

declared="$WORK/declared.paths"
: >"$declared"
dest_map="$WORK/dest.map"
: >"$dest_map"

while IFS="$(printf '\t')" read -r rel expected klass; do
  [ -n "$rel" ] || continue
  printf '%s\n' "$rel" >>"$declared"
  if [ "$klass" = "sidecar" ]; then
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
  fi
  dest_file="$DEST/$rel"
  if [ -L "$dest_file" ]; then
    fail dest-symlink
  fi
  if [ ! -f "$dest_file" ]; then
    if [ "$klass" = "sidecar" ]; then
      fail missing-sidecar
    fi
    fail missing-artifact
  fi
  actual=$(sha256_file "$dest_file")
  if [ "$actual" != "$expected" ]; then
    fail altered-artifact
  fi
  printf '%s\t%s\n' "$rel" "$actual" >>"$dest_map"
done < <(cat "$WORK/artifacts.tsv" "$WORK/sidecars.tsv")

while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  if ! grep -Fxq "$rel" "$declared"; then
    fail extra-artifact
  fi
done < <(find -P "$DEST" -type f -print 2>/dev/null | sed "s|^$DEST/||")

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

printf 'MODEL_VERIFY_OK count=%s sidecar_count=%s dest_sha256=%s\n' \
  "$count" "$sidecar_count" "$dest_sha256"
