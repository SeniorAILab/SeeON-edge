#include "native_perception.hpp"
#include "preprocess_cpu.hpp"
#include "preprocess_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::array<std::array<int, 2>, 6> kGeometries{{
    {1080, 1920},
    {720, 1280},
    {360, 640},
    {102, 100},
    {640, 640},
    {576, 720},
}};
constexpr std::array<std::uint32_t, 3> kSeeds{{20260901U, 20260902U, 20260903U}};
constexpr int kRuns = 2;
constexpr std::uint8_t kPaddingSentinel = 0xA5;
constexpr std::size_t kMatrixCases = kGeometries.size() * kSeeds.size() * 2;

bool cuda_ok(cudaError_t result, const char* action) {
  if (result == cudaSuccess) return true;
  std::fprintf(stderr, "preprocess_gpu_test: %s: %s\n", action, cudaGetErrorString(result));
  return false;
}

void fill_rgba(std::vector<std::uint8_t>* rgba, int width, int height, std::size_t pitch,
               std::uint32_t seed) {
  std::fill(rgba->begin(), rgba->end(), kPaddingSentinel);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y) * pitch +
                                 static_cast<std::size_t>(x) * 4;
      (*rgba)[offset] = static_cast<std::uint8_t>((seed + x * 17U + y * 31U) & 0xffU);
      (*rgba)[offset + 1] =
          static_cast<std::uint8_t>((seed * 3U + x * 29U + y * 7U) & 0xffU);
      (*rgba)[offset + 2] =
          static_cast<std::uint8_t>((seed * 5U + x * 11U + y * 43U) & 0xffU);
      (*rgba)[offset + 3] =
          static_cast<std::uint8_t>((seed * 7U + x * 19U + y * 23U) & 0xffU);
    }
  }
}

bool assert_padding(const std::vector<float>& tensor,
                    const seeon::perception::AffineMetadata& affine) {
  const float expected = static_cast<float>(seeon::perception::kLetterboxPadValue) / 255.0F;
  const std::size_t plane =
      static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
  for (int y = 0; y < affine.tensor_height; ++y) {
    for (int x = 0; x < affine.tensor_width; ++x) {
      if (x >= affine.pad_left() && x < affine.pad_left() + affine.content_width &&
          y >= affine.pad_top() && y < affine.pad_top() + affine.content_height) {
        continue;
      }
      const std::size_t offset = static_cast<std::size_t>(y) * affine.tensor_width + x;
      for (int channel = 0; channel < 3; ++channel) {
        if (std::memcmp(&tensor[static_cast<std::size_t>(channel) * plane + offset], &expected,
                        sizeof(expected)) != 0) {
          std::fprintf(stderr, "preprocess_gpu_test: pad mismatch x=%d y=%d channel=%d\n", x, y,
                       channel);
          return false;
        }
      }
    }
  }
  return true;
}

bool assert_bgr_semantics(const std::vector<std::uint8_t>& rgba, std::size_t pitch, int width,
                          int height, const seeon::perception::AffineMetadata& affine,
                          const std::vector<float>& tensor) {
  const int x = affine.pad_left() + affine.content_width / 2;
  const int y = affine.pad_top() + affine.content_height / 2;
  const int content_x = x - affine.pad_left();
  const int content_y = y - affine.pad_top();
  const double source_x =
      (content_x + 0.5) * static_cast<double>(width) / affine.content_width - 0.5;
  const double source_y =
      (content_y + 0.5) * static_cast<double>(height) / affine.content_height - 0.5;
  const double clamped_x = std::max(0.0, std::min(source_x, static_cast<double>(width - 1)));
  const double clamped_y = std::max(0.0, std::min(source_y, static_cast<double>(height - 1)));
  const int x0 = static_cast<int>(clamped_x);
  const int x1 = std::min(x0 + 1, width - 1);
  const int y0 = static_cast<int>(clamped_y);
  const int y1 = std::min(y0 + 1, height - 1);
  const double wx = clamped_x - x0;
  const double wy = clamped_y - y0;
  const std::size_t plane =
      static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
  const std::size_t output_offset = static_cast<std::size_t>(y) * affine.tensor_width + x;
  for (int channel = 0; channel < 3; ++channel) {
    const int source_byte = 2 - channel;
    const auto pixel = [&](int source_y_index, int source_x_index) {
      return rgba[static_cast<std::size_t>(source_y_index) * pitch +
                  static_cast<std::size_t>(source_x_index) * 4 + source_byte];
    };
    const double top = (1.0 - wx) * pixel(y0, x0) + wx * pixel(y0, x1);
    const double bottom = (1.0 - wx) * pixel(y1, x0) + wx * pixel(y1, x1);
    const float expected = static_cast<float>(((1.0 - wy) * top + wy * bottom) / 255.0);
    if (std::memcmp(&tensor[static_cast<std::size_t>(channel) * plane + output_offset],
                    &expected, sizeof(expected)) != 0) {
      std::fprintf(stderr, "preprocess_gpu_test: BGR semantic mismatch x=%d y=%d channel=%d\n",
                   x, y, channel);
      return false;
    }
  }
  return true;
}

bool compare_bytes(const std::vector<float>& expected, const std::vector<float>& actual,
                   const char* comparison, int height, int width, std::uint32_t seed,
                   std::size_t pitch, int run) {
  const std::size_t bytes = expected.size() * sizeof(float);
  const auto* expected_bytes = reinterpret_cast<const std::uint8_t*>(expected.data());
  const auto* actual_bytes = reinterpret_cast<const std::uint8_t*>(actual.data());
  if (std::memcmp(expected_bytes, actual_bytes, bytes) == 0) return true;
  std::size_t mismatch = 0;
  while (expected_bytes[mismatch] == actual_bytes[mismatch]) ++mismatch;
  std::fprintf(stderr,
               "preprocess_gpu_test: %s mismatch height=%d width=%d seed=%u pitch=%zu run=%d "
               "byte=%zu expected=%02x actual=%02x\n",
               comparison, height, width, seed, pitch, run, mismatch, expected_bytes[mismatch],
               actual_bytes[mismatch]);
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "enumerating CUDA devices") || device_count < 1 ||
      !cuda_ok(cudaSetDevice(0), "selecting CUDA device")) {
    return EXIT_FAILURE;
  }

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating CUDA stream")) {
    return EXIT_FAILURE;
  }

  bool success = true;
  std::vector<std::vector<float>> baselines(kGeometries.size() * kSeeds.size() * 2);
  std::size_t case_count = 0;
  for (int run = 0; success && run < kRuns; ++run) {
    std::size_t baseline_index = 0;
    for (const auto& geometry : kGeometries) {
      const int height = geometry[0];
      const int width = geometry[1];
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const std::size_t tensor_elements =
          3ULL * static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
      for (const std::uint32_t seed : kSeeds) {
        for (const std::size_t pitch : {static_cast<std::size_t>(width) * 4,
                                        static_cast<std::size_t>(width) * 4 + 64}) {
          std::vector<std::uint8_t> host_rgba(pitch * static_cast<std::size_t>(height));
          fill_rgba(&host_rgba, width, height, pitch, seed);
          std::vector<float> cpu_tensor(tensor_elements);
          std::vector<float> gpu_tensor(tensor_elements);
          seeon::trt::preprocess_rgba_to_bgr_tensor(host_rgba.data(), width, height,
                                                     static_cast<int>(pitch), affine,
                                                     cpu_tensor.data());

          std::uint8_t* device_rgba = nullptr;
          float* device_tensor = nullptr;
          if (!cuda_ok(cudaMalloc(&device_rgba, host_rgba.size()), "allocating pitched RGBA") ||
              !cuda_ok(cudaMalloc(&device_tensor, tensor_elements * sizeof(float)),
                       "allocating tensor") ||
              !cuda_ok(cudaMemcpy2DAsync(device_rgba, pitch, host_rgba.data(), pitch, pitch,
                                         static_cast<std::size_t>(height),
                                         cudaMemcpyHostToDevice, stream),
                       "copying pitched RGBA to device")) {
            if (device_tensor != nullptr) cudaFree(device_tensor);
            if (device_rgba != nullptr) cudaFree(device_rgba);
            success = false;
            break;
          }

          std::string error;
          if (!seeon::trt::preprocess_rgba_device_to_bgr_tensor(
                  device_rgba, width, height, pitch, affine, device_tensor, stream, &error)) {
            std::fprintf(stderr, "preprocess_gpu_test: production preprocess failed: %s\n",
                         error.c_str());
            cudaFree(device_tensor);
            cudaFree(device_rgba);
            success = false;
            break;
          }
          if (!cuda_ok(cudaMemcpyAsync(gpu_tensor.data(), device_tensor,
                                       tensor_elements * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream),
                       "copying full tensor to host") ||
              !cuda_ok(cudaStreamSynchronize(stream), "synchronizing CUDA stream") ||
              !cuda_ok(cudaFree(device_tensor), "freeing tensor") ||
              !cuda_ok(cudaFree(device_rgba), "freeing pitched RGBA")) {
            success = false;
            break;
          }

          if (!compare_bytes(cpu_tensor, gpu_tensor, "CPU/GPU", height, width, seed, pitch, run) ||
              !assert_padding(gpu_tensor, affine) ||
              !assert_bgr_semantics(host_rgba, pitch, width, height, affine, gpu_tensor)) {
            success = false;
            break;
          }
          if (run == 0) {
            baselines[baseline_index] = gpu_tensor;
          } else if (!compare_bytes(baselines[baseline_index], gpu_tensor, "determinism", height,
                                    width, seed, pitch, run)) {
            success = false;
            break;
          }
          ++case_count;
          ++baseline_index;
        }
        if (!success) break;
      }
      if (!success) break;
    }
  }

  if (!cuda_ok(cudaStreamDestroy(stream), "destroying CUDA stream")) success = false;
  if (!success) return EXIT_FAILURE;
  std::printf("PREPROCESS_GPU_PARITY_RECEIPT geometries=6 seeds=3 pitches=2 cases=%zu runs=%d "
              "production_kernel_executions=%zu deterministic=1 bit_exact=1\n",
              kMatrixCases, kRuns, case_count);
  return EXIT_SUCCESS;
}
