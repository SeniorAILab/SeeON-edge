#pragma once

#include "native_perception.hpp"

#include <cstdint>

namespace seeon::trt {

// CPU preprocessing pinned to the shipped inference path: double bilinear
// resize, value-114 padding, RGBA -> BGR planar FP32 normalization.
void preprocess_rgba_to_bgr_tensor(const std::uint8_t* rgba, int width, int height,
                                   int stride, const perception::AffineMetadata& affine,
                                   float* output_chw);

}  // namespace seeon::trt
