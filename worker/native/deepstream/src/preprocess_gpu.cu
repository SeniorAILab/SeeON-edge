#include "preprocess_gpu.hpp"

#include <cuda_runtime_api.h>

namespace seeon::trt {
namespace {

__global__ void preprocess_kernel(const std::uint8_t* rgba, int width, int height,
                                  std::size_t pitch_bytes, int tensor_width,
                                  int tensor_height, int content_width,
                                  int content_height, int pad_left, int pad_top,
                                  float* output_chw) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= tensor_width || y >= tensor_height) return;

  const std::size_t plane = static_cast<std::size_t>(tensor_width) * tensor_height;
  const std::size_t output_offset = static_cast<std::size_t>(y) * tensor_width + x;
  if (x < pad_left || x >= pad_left + content_width || y < pad_top ||
      y >= pad_top + content_height) {
    constexpr float kPad = static_cast<float>(perception::kLetterboxPadValue) / 255.0F;
    output_chw[output_offset] = kPad;
    output_chw[plane + output_offset] = kPad;
    output_chw[2 * plane + output_offset] = kPad;
    return;
  }

  const int content_x = x - pad_left;
  const int content_y = y - pad_top;
  const double scale_x = static_cast<double>(width) / content_width;
  const double scale_y = static_cast<double>(height) / content_height;
  const double source_x = (content_x + 0.5) * scale_x - 0.5;
  const double source_y = (content_y + 0.5) * scale_y - 0.5;
  const double clamped_x = max(0.0, min(source_x, static_cast<double>(width - 1)));
  const double clamped_y = max(0.0, min(source_y, static_cast<double>(height - 1)));
  const int x0 = static_cast<int>(clamped_x);
  const int x1 = min(x0 + 1, width - 1);
  const int y0 = static_cast<int>(clamped_y);
  const int y1 = min(y0 + 1, height - 1);
  const double wx = clamped_x - x0;
  const double wy = clamped_y - y0;
  const std::uint8_t* row0 = rgba + static_cast<std::size_t>(y0) * pitch_bytes;
  const std::uint8_t* row1 = rgba + static_cast<std::size_t>(y1) * pitch_bytes;

  for (int channel = 0; channel < 3; ++channel) {
    const int source_byte = 2 - channel;
    const double top =
        (1.0 - wx) * row0[static_cast<std::size_t>(x0) * 4 + source_byte] +
        wx * row0[static_cast<std::size_t>(x1) * 4 + source_byte];
    const double bottom =
        (1.0 - wx) * row1[static_cast<std::size_t>(x0) * 4 + source_byte] +
        wx * row1[static_cast<std::size_t>(x1) * 4 + source_byte];
    const double value = (1.0 - wy) * top + wy * bottom;
    output_chw[static_cast<std::size_t>(channel) * plane + output_offset] =
        static_cast<float>(value / 255.0);
  }
}

}  // namespace

bool preprocess_rgba_device_to_bgr_tensor(
    const void* rgba_device, int width, int height, std::size_t pitch_bytes,
    const perception::AffineMetadata& affine, float* output_chw, cudaStream_t stream,
    std::string* error) {
  if (rgba_device == nullptr) {
    *error = "preprocess_device_pointer_invalid";
    return false;
  }
  if (width <= 0 || height <= 0 || affine.tensor_width <= 0 || affine.tensor_height <= 0 ||
      affine.content_width <= 0 || affine.content_height <= 0) {
    *error = "preprocess_geometry_invalid";
    return false;
  }
  if (pitch_bytes < static_cast<std::size_t>(width) * 4) {
    *error = "preprocess_pitch_invalid";
    return false;
  }
  if (output_chw == nullptr) {
    *error = "preprocess_destination_invalid";
    return false;
  }
  if (stream == nullptr) {
    *error = "preprocess_stream_invalid";
    return false;
  }

  constexpr dim3 block{16, 16};
  const dim3 grid{static_cast<unsigned int>((affine.tensor_width + block.x - 1) / block.x),
                  static_cast<unsigned int>((affine.tensor_height + block.y - 1) / block.y)};
  // Discard an unrelated asynchronous error so the result below belongs to
  // this launch.
  (void)cudaGetLastError();
  preprocess_kernel<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(rgba_device),
                                                  width, height, pitch_bytes,
                                                  affine.tensor_width, affine.tensor_height,
                                                  affine.content_width, affine.content_height,
                                                  affine.pad_left(), affine.pad_top(), output_chw);
  if (cudaGetLastError() != cudaSuccess) {
    *error = "preprocess_kernel_launch_failed";
    return false;
  }
  return true;
}

}  // namespace seeon::trt
