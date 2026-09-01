#include "postprocess_gpu.hpp"

#include <cuda_runtime_api.h>

namespace seeon::trt {
namespace {

template <int kStride, bool kPose>
__global__ void compact_rows_kernel(const float* source_rows, float* compact_rows,
                                    PostprocessChannelHeader* header) {
  int count = 0;
  const auto* source_words = reinterpret_cast<const unsigned int*>(source_rows);
  auto* compact_words = reinterpret_cast<unsigned int*>(compact_rows);
  for (int row = 0; row < kPostprocessTensorRows; ++row) {
    const float score = source_rows[row * kStride + 4];
    const bool keep = kPose
                          ? static_cast<double>(score) > 0.05
                          : (static_cast<int>(static_cast<double>(
                                 source_rows[row * kStride + 5])) == 0 &&
                             !(static_cast<double>(score) < 0.25));
    if (!keep) continue;
    const int source_offset = row * kStride;
    const int compact_offset = count * kStride;
    for (int column = 0; column < kStride; ++column) {
      compact_words[compact_offset + column] = source_words[source_offset + column];
    }
    ++count;
  }
  header->kernel_executed = 1;
  header->count = count;
}

bool validate_arguments(const float* source_rows, float* compact_rows,
                        PostprocessChannelHeader* header, cudaStream_t stream,
                        std::string* error) {
  if (source_rows == nullptr || compact_rows == nullptr || header == nullptr) {
    *error = "postprocess_destination_invalid";
    return false;
  }
  if (stream == nullptr) {
    *error = "postprocess_stream_invalid";
    return false;
  }
  return true;
}

template <int kStride, bool kPose>
bool launch_compaction(const float* source_rows, float* compact_rows,
                       PostprocessChannelHeader* header, cudaStream_t stream,
                       std::string* error) {
  if (!validate_arguments(source_rows, compact_rows, header, stream, error)) return false;
  (void)cudaGetLastError();
  compact_rows_kernel<kStride, kPose><<<1, 1, 0, stream>>>(source_rows, compact_rows, header);
  if (cudaGetLastError() != cudaSuccess) {
    *error = "postprocess_kernel_launch_failed";
    return false;
  }
  return true;
}

}  // namespace

bool compact_pose_rows_device(const float* source_rows, float* compact_rows,
                              PostprocessChannelHeader* header, cudaStream_t stream,
                              std::string* error) {
  return launch_compaction<kPostprocessPoseRowStride, true>(source_rows, compact_rows, header,
                                                             stream, error);
}

bool compact_person_rows_device(const float* source_rows, float* compact_rows,
                                PostprocessChannelHeader* header, cudaStream_t stream,
                                std::string* error) {
  return launch_compaction<kPostprocessPersonRowStride, false>(
      source_rows, compact_rows, header, stream, error);
}

}  // namespace seeon::trt
