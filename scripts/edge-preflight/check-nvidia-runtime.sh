#!/bin/sh
set -eu

prefix='[edge-preflight]'

die() {
  printf '%s %s\n' "$prefix" "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_command docker
require_command nvidia-smi
require_command nvidia-ctk
require_command nvidia-container-runtime

docker info >/dev/null 2>&1 || die "Docker daemon is not reachable"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not available"
nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver is not healthy"
nvidia-container-runtime --version >/dev/null 2>&1 || die "nvidia-container-runtime is not executable"

runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null) || die "Docker runtime list is unavailable"
if ! printf '%s' "$runtimes" | grep '"nvidia"' >/dev/null 2>&1; then
  die "Docker NVIDIA runtime is not registered. Run on the edge host: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

printf '%s Docker NVIDIA runtime is registered; image-backed compose can start the ml-worker GPU reservation.\n' "$prefix"
