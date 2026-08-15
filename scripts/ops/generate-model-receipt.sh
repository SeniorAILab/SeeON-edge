#!/bin/bash
# Pre-boot private hash-only receipt generator for the model materialization gate.
#
# Root-only wrapper. Delegates to generate-model-receipt.py so effective UID
# and atomic --out publication cannot be spoofed through PATH. Inspects an
# approved running legacy worker through the normal Docker CLI, hashes the
# closed default artifact set plus tracked model-root sidecars, and writes
# the receipt consumed by materialize-model-artifacts.sh. Emits only a
# count/sidecar_count/dest_sha256 verdict. Never prints credentials, RTSP,
# model bytes, source identity, or paths.
set -euo pipefail
set +x
umask 077

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$SCRIPT_DIR/generate-model-receipt.py" "$@"
