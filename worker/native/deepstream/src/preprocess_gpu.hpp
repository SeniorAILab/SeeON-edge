#pragma once

#include "native_perception.hpp"

#include <cstddef>
#include <string>

#include <cuda_runtime_api.h>

namespace seeon::trt {

// Attempts to enqueue RGBA pitched-device -> BGR planar FP32 letterbox
// preprocessing on stream. The source and destination are borrowed device
// allocations; this function never stages pixels through host or unified
// memory. Once called, the caller must synchronize stream before releasing
// either borrowed allocation, including when this returns false.
[[nodiscard]] bool preprocess_rgba_device_to_bgr_tensor(
    const void* rgba_device, int width, int height, std::size_t pitch_bytes,
    const perception::AffineMetadata& affine, float* output_chw, cudaStream_t stream,
    std::string* error);

}  // namespace seeon::trt
