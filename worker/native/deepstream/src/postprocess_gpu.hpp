#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>

namespace seeon::trt {

struct PostprocessChannelHeader {
  std::int32_t kernel_executed;
  std::int32_t count;
};
static_assert(sizeof(PostprocessChannelHeader) == 8);

inline constexpr int kPostprocessTensorRows = 300;
inline constexpr int kPostprocessPoseRowStride = 57;
inline constexpr int kPostprocessPersonRowStride = 6;

// Enqueues source-order filtering and full-row byte copies. The caller owns all
// buffers through stream completion.
[[nodiscard]] bool compact_pose_rows_device(
    const float* source_rows, float* compact_rows, PostprocessChannelHeader* header,
    cudaStream_t stream, std::string* error);
[[nodiscard]] bool compact_person_rows_device(
    const float* source_rows, float* compact_rows, PostprocessChannelHeader* header,
    cudaStream_t stream, std::string* error);

}  // namespace seeon::trt
