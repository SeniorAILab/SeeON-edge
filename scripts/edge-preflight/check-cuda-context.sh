#!/bin/sh
# Preflight: verify the CUDA *context* actually initializes, not just that
# nvidia-smi/NVML enumerate the GPU. Catches the ADR-0002 second-layer failure
# (driver 580.x + CUDA 13.0 -> cuInit(0)=CUDA_ERROR_NO_DEVICE) BEFORE the worker
# starts and silently degrades to CPU. Fail-loud, matching the runtime policy.
set -eu

prefix='[edge-preflight]'
die() { printf '%s %s\n' "$prefix" "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "missing nvidia-smi"
command -v python3 >/dev/null 2>&1 || die "missing python3"
nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver not healthy (nvidia-smi failed)"

# Driver-level cuInit probe (libcuda). rc 0 = OK; 100 = CUDA_ERROR_NO_DEVICE.
rc=$(python3 - <<'PY'
import ctypes, sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError:
    print("no-libcuda"); sys.exit(0)
print(lib.cuInit(0))
PY
)
case "$rc" in
  0) : ;;
  no-libcuda) die "libcuda.so.1 not found in the container/host" ;;
  100) die "cuInit(0)=100 (CUDA_ERROR_NO_DEVICE): CUDA context cannot init even though nvidia-smi works. Known driver/CUDA-pairing breakage (see ADR-0002 + docs/runbooks/driver-cuda-alignment.md). Align the host driver/CUDA before starting the GPU worker." ;;
  *) die "cuInit(0)=$rc (not 0): CUDA driver API init failed; see docs/runbooks/driver-cuda-alignment.md" ;;
esac

# torch-level check (arch kernels + availability), when torch is importable.
python3 - <<'PY' || exit 1
import sys
try:
    import torch
except Exception:
    print("[edge-preflight] torch not importable in this context; skipping torch check")
    sys.exit(0)
archs = list(torch.cuda.get_arch_list())
if not archs:
    sys.exit("[edge-preflight] torch has an EMPTY arch_list (no compiled GPU kernels); pin a CUDA wheel with your GPU's sm_XXX (ADR-0002)")
if not torch.cuda.is_available():
    sys.exit(f"[edge-preflight] torch.cuda.is_available()=False despite arch_list={archs}; driver/CUDA pairing issue (ADR-0002)")
print(f"[edge-preflight] CUDA context OK: cuInit=0, torch {torch.__version__}, arch_list={archs}")
PY

printf '%s CUDA context healthy (cuInit=0, torch CUDA available).\n' "$prefix"
