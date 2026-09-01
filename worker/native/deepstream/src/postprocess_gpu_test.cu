#include "postprocess_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using seeon::trt::PostprocessChannelHeader;

bool cuda_ok(cudaError_t status, const char* action) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "postprocess_gpu_test: %s: %s\n", action, cudaGetErrorString(status));
  return false;
}

void fill_rows(std::vector<float>* rows, int stride) {
  auto* words = reinterpret_cast<std::uint32_t*>(rows->data());
  for (std::size_t index = 0; index < rows->size(); ++index) {
    words[index] = 0x3f000000U ^ static_cast<std::uint32_t>(index * 2654435761U);
  }
  for (int row = 0; row < seeon::trt::kPostprocessTensorRows; ++row) {
    (*rows)[row * stride + 4] = 0.0F;
    if (stride == seeon::trt::kPostprocessPersonRowStride) (*rows)[row * stride + 5] = 1.0F;
  }
}

struct RunResult {
  PostprocessChannelHeader header{};
  std::vector<float> rows;
};

using CompactFunction = bool (*)(const float*, float*, PostprocessChannelHeader*, cudaStream_t,
                                 std::string*);

bool run_compaction(const std::vector<float>& source, int stride, CompactFunction compact,
                    RunResult* result, std::size_t* transferred_row_bytes) {
  float* device_source = nullptr;
  float* device_compact = nullptr;
  PostprocessChannelHeader* device_header = nullptr;
  cudaStream_t stream = nullptr;
  bool success = cuda_ok(cudaStreamCreate(&stream), "creating stream") &&
                 cuda_ok(cudaMalloc(&device_source, source.size() * sizeof(float)),
                         "allocating source") &&
                 cuda_ok(cudaMalloc(&device_compact, source.size() * sizeof(float)),
                         "allocating compact") &&
                 cuda_ok(cudaMalloc(&device_header, sizeof(*device_header)), "allocating header") &&
                 cuda_ok(cudaMemcpyAsync(device_source, source.data(), source.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream), "copying source");
  std::string error;
  if (success && !compact(device_source, device_compact, device_header, stream, &error)) {
    std::fprintf(stderr, "postprocess_gpu_test: production compaction failed: %s\n", error.c_str());
    success = false;
  }
  if (success) {
    success = cuda_ok(cudaMemcpyAsync(&result->header, device_header, sizeof(result->header),
                                      cudaMemcpyDeviceToHost, stream), "copying header") &&
              cuda_ok(cudaStreamSynchronize(stream), "synchronizing header");
  }
  if (success && (result->header.kernel_executed != 1 || result->header.count < 0 ||
                  result->header.count > seeon::trt::kPostprocessTensorRows)) {
    std::fprintf(stderr, "postprocess_gpu_test: invalid header executed=%d count=%d\n",
                 result->header.kernel_executed, result->header.count);
    success = false;
  }
  if (success) {
    result->rows.resize(static_cast<std::size_t>(result->header.count) * stride);
    const std::size_t bytes = result->rows.size() * sizeof(float);
    if (bytes != 0) {
      success = cuda_ok(cudaMemcpyAsync(result->rows.data(), device_compact, bytes,
                                        cudaMemcpyDeviceToHost, stream), "copying survivor rows");
    }
    success = success && cuda_ok(cudaStreamSynchronize(stream), "synchronizing survivor rows");
    *transferred_row_bytes += bytes;
  }
  if (device_header != nullptr) cudaFree(device_header);
  if (device_compact != nullptr) cudaFree(device_compact);
  if (device_source != nullptr) cudaFree(device_source);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return success;
}

bool expect_rows(const std::vector<float>& source, int stride, const std::vector<int>& expected,
                 const RunResult& actual, const char* label) {
  if (actual.header.count != static_cast<int>(expected.size())) {
    std::fprintf(stderr, "postprocess_gpu_test: %s count mismatch expected=%zu actual=%d\n", label,
                 expected.size(), actual.header.count);
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float* expected_row = source.data() + static_cast<std::size_t>(expected[index]) * stride;
    const float* actual_row = actual.rows.data() + index * stride;
    if (std::memcmp(expected_row, actual_row, static_cast<std::size_t>(stride) * sizeof(float)) != 0) {
      std::fprintf(stderr, "postprocess_gpu_test: %s byte/source-order mismatch survivor=%zu source=%d\n",
                   label, index, expected[index]);
      return false;
    }
  }
  return true;
}

bool run_case(std::vector<float> source, int stride, CompactFunction compact,
              const std::vector<int>& expected, const char* label,
              std::size_t* transferred_row_bytes) {
  RunResult first;
  RunResult second;
  return run_compaction(source, stride, compact, &first, transferred_row_bytes) &&
         expect_rows(source, stride, expected, first, label) &&
         run_compaction(source, stride, compact, &second, transferred_row_bytes) &&
         first.header.kernel_executed == second.header.kernel_executed &&
         first.header.count == second.header.count &&
         (first.rows.empty() ||
          std::memcmp(first.rows.data(), second.rows.data(),
                      first.rows.size() * sizeof(float)) == 0);
}

bool pose_cases(std::size_t* transferred_row_bytes) {
  constexpr int stride = seeon::trt::kPostprocessPoseRowStride;
  std::vector<float> rows(seeon::trt::kPostprocessTensorRows * stride);
  fill_rows(&rows, stride);
  rows[0 * stride + 4] = std::nextafter(0.05F, 0.0F);
  rows[1 * stride + 4] = 0.05F;
  rows[2 * stride + 4] = std::nextafter(0.05F, 1.0F);
  rows[5 * stride + 4] = 0.8F;
  rows[8 * stride + 4] = 0.9F;
  // 0.05 is not exactly representable as float. Promoting 0.05F to double
  // yields a value just above the double threshold, matching the CPU parser.
  if (!run_case(rows, stride, seeon::trt::compact_pose_rows_device, {1, 2, 5, 8},
                "pose-boundary-mixed", transferred_row_bytes)) return false;
  fill_rows(&rows, stride);
  if (!run_case(rows, stride, seeon::trt::compact_pose_rows_device, {}, "pose-zero",
                transferred_row_bytes)) return false;
  for (int row = 0; row < seeon::trt::kPostprocessTensorRows; ++row) rows[row * stride + 4] = 0.6F;
  return run_case(rows, stride, seeon::trt::compact_pose_rows_device,
                  [] { std::vector<int> all(seeon::trt::kPostprocessTensorRows); for (int i = 0; i < static_cast<int>(all.size()); ++i) all[i] = i; return all; }(),
                  "pose-all", transferred_row_bytes);
}

bool person_cases(std::size_t* transferred_row_bytes) {
  constexpr int stride = seeon::trt::kPostprocessPersonRowStride;
  std::vector<float> rows(seeon::trt::kPostprocessTensorRows * stride);
  fill_rows(&rows, stride);
  rows[0 * stride + 4] = 0.9F; rows[0 * stride + 5] = 1.0F;
  rows[1 * stride + 4] = std::nextafter(0.25F, 0.0F); rows[1 * stride + 5] = 0.0F;
  rows[2 * stride + 4] = 0.25F; rows[2 * stride + 5] = 0.0F;
  rows[3 * stride + 4] = std::nextafter(0.25F, 1.0F); rows[3 * stride + 5] = 0.0F;
  rows[7 * stride + 4] = std::numeric_limits<float>::quiet_NaN(); rows[7 * stride + 5] = 0.0F;
  if (!run_case(rows, stride, seeon::trt::compact_person_rows_device, {2, 3, 7},
                "person-class-boundary-mixed", transferred_row_bytes)) return false;
  fill_rows(&rows, stride);
  if (!run_case(rows, stride, seeon::trt::compact_person_rows_device, {}, "person-zero",
                transferred_row_bytes)) return false;
  for (int row = 0; row < seeon::trt::kPostprocessTensorRows; ++row) {
    rows[row * stride + 4] = 0.6F;
    rows[row * stride + 5] = 0.0F;
  }
  std::vector<int> all(seeon::trt::kPostprocessTensorRows);
  for (int row = 0; row < static_cast<int>(all.size()); ++row) all[row] = row;
  return run_case(rows, stride, seeon::trt::compact_person_rows_device, all, "person-all",
                  transferred_row_bytes);
}

}  // namespace

int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "enumerating CUDA devices") || device_count < 1 ||
      !cuda_ok(cudaSetDevice(0), "selecting CUDA device")) return EXIT_FAILURE;
  std::size_t row_bytes = 0;
  if (!pose_cases(&row_bytes) || !person_cases(&row_bytes)) return EXIT_FAILURE;
  constexpr std::size_t kHeaderCopies = 12;
  std::printf("POSTPROCESS_GPU_RECEIPT pose_cases=3 person_cases=3 deterministic_reruns=1 "
              "headers_executed=1 d2h_header_bytes=%zu d2h_row_bytes=%zu\n",
              kHeaderCopies * sizeof(PostprocessChannelHeader), row_bytes);
  return EXIT_SUCCESS;
}
