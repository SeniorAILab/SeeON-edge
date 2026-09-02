/*
 * Private glibc-2.39 x86_64-FMA atan2 compatibility surface.
 *
 * Derived from the IBM Accurate Mathematical Library sources listed in
 * pinned_host_atan2.cu.  Copyright (C) 2001-2024 Free Software Foundation,
 * Inc.  Licensed under LGPL-2.1-or-later; this notice is retained with the
 * derived implementation. Modified by SeniorAILab on 2026-09-01 for the
 * finite CUDA contour domain.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

// Only for finite double contour deltas in [-159, 159].  This is deliberately
// not a general device math API.
__device__ double pinned_host_atan2(double y, double x);

#if defined(SEEON_PINNED_HOST_ATAN2_TEST)
// Private test seam: copies the exact device constant table without exposing
// it to production callers.
cudaError_t pinned_host_atan2_copy_table(uint64_t table[241][7]);
#endif
