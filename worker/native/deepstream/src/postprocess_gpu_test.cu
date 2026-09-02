#include "native_perception.hpp"
#include "postprocess_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using seeon::perception::AffineMetadata;
using seeon::perception::ParsedBedRegion;
using seeon::trt::BedFinalizeGeometry;
using seeon::trt::BedFinalizeWorkspace;
using seeon::trt::PackedBedRecord;
using seeon::trt::PostprocessChannelHeader;

static_assert(sizeof(PackedBedRecord) == 416);
static_assert(seeon::trt::kPostprocessBedChannelMaxBytes == 124808);
static_assert(seeon::trt::kPostprocessMaxPersonTransferBytes == 200424);
static_assert(seeon::trt::kPostprocessPoseOnlyTransferBytes == 193224);

bool validate_headers(const PostprocessChannelHeader& pose, const PostprocessChannelHeader& person,
                      const PostprocessChannelHeader& bed, bool person_required,
                      bool expected_valid, const char* label) {
  std::string error;
  const bool valid = seeon::trt::validate_postprocess_channel_headers(
      pose, person, bed, person_required, &error);
  if (valid != expected_valid ||
      (!valid && error != "postprocess_header_invalid")) {
    std::fprintf(stderr, "postprocess_gpu_test: header case %s valid=%d error=%s\n", label,
                 valid, error.c_str());
    return false;
  }
  return true;
}

bool verify_header_validation() {
  const PostprocessChannelHeader valid{1, 0};
  const PostprocessChannelHeader optional{0, 0};
  const std::array<PostprocessChannelHeader, 4> invalid_required = {
      PostprocessChannelHeader{0, 0}, PostprocessChannelHeader{1, -1},
      PostprocessChannelHeader{1, 301}, PostprocessChannelHeader{2, 0}};
  bool ok = validate_headers(valid, optional, valid, false, true, "optional-0-0");
  ok = ok && validate_headers(PostprocessChannelHeader{0, 0}, optional, valid, false, false,
                              "pose-executed-0");
  ok = ok && validate_headers(valid, optional, PostprocessChannelHeader{0, 0}, false, false,
                              "bed-executed-0");
  ok = ok && validate_headers(valid, PostprocessChannelHeader{0, 0}, valid, true, false,
                              "person-required-0-0");
  for (const PostprocessChannelHeader& header : invalid_required) {
    ok = ok && validate_headers(header, optional, valid, false, false, "pose-required-matrix");
    ok = ok && validate_headers(valid, header, valid, true, false, "person-required-matrix");
    ok = ok && validate_headers(valid, optional, header, false, false, "bed-required-matrix");
  }
  for (const PostprocessChannelHeader& header :
       {PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{2, 0},
        PostprocessChannelHeader{0, 1},
        PostprocessChannelHeader{0, -1}, PostprocessChannelHeader{0, 301}}) {
    ok = ok && validate_headers(valid, header, valid, false, false, "optional-person-not-0-0");
  }
  return ok;
}

bool verify_invalid_headers_skip_metadata_conversion() {
  struct HeaderCase {
    std::array<PostprocessChannelHeader, 3> headers;
    bool person_required;
  };
  const std::array<HeaderCase, 9> invalid_headers = {{
      {{PostprocessChannelHeader{0, 0}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, 0}}, false},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{0, 0}}, false},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, 0}}, true},
      {{PostprocessChannelHeader{1, -1}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, 0}}, false},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{1, -1},
        PostprocessChannelHeader{1, 0}}, true},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, -1}}, false},
      {{PostprocessChannelHeader{1, 301}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, 0}}, false},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{1, 301},
        PostprocessChannelHeader{1, 0}}, true},
      {{PostprocessChannelHeader{1, 0}, PostprocessChannelHeader{0, 0},
        PostprocessChannelHeader{1, 301}}, false},
  }};
  for (const HeaderCase& header_case : invalid_headers) {
    const auto& headers = header_case.headers;
    std::string error;
    std::size_t metadata_bytes = 0;
    bool converted = false;
    if (seeon::trt::validate_postprocess_channel_headers(
            headers[0], headers[1], headers[2], header_case.person_required, &error)) {
      converted = seeon::trt::postprocess_transfer_bytes(
          static_cast<std::size_t>(headers[0].count),
          static_cast<std::size_t>(headers[1].count),
          static_cast<std::size_t>(headers[2].count), &metadata_bytes);
    }
    if (converted || metadata_bytes != 0 || error != "postprocess_header_invalid") {
      std::fprintf(stderr, "postprocess_gpu_test: invalid header converted metadata\n");
      return false;
    }
  }
  return true;
}

bool cuda_ok(cudaError_t status, const char* action) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "postprocess_gpu_test: %s: %s\n", action, cudaGetErrorString(status));
  return false;
}

template <typename T>
class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }

  bool allocate(std::size_t count, const char* label) {
    return cuda_ok(cudaMalloc(&pointer_, count * sizeof(T)), label);
  }
  [[nodiscard]] T* get() const { return pointer_; }
  [[nodiscard]] std::size_t bytes(std::size_t count) const { return count * sizeof(T); }

 private:
  T* pointer_ = nullptr;
};

class BedWorkspace {
 public:
  bool allocate() {
    constexpr std::size_t entries = seeon::trt::kBedFinalizeEntries;
    constexpr std::size_t segments = seeon::trt::kBedFinalizeSegments;
    if (!keys_[0].allocate(entries, "allocating bed keys") ||
        !keys_[1].allocate(entries, "allocating bed keys") ||
        !points_[0].allocate(entries, "allocating bed points") ||
        !points_[1].allocate(entries, "allocating bed points") ||
        !row_to_record_.allocate(segments, "allocating bed row offsets") ||
        !active_count_.allocate(segments, "allocating bed active counts") ||
        !crop_.allocate(segments, "allocating bed crops") ||
        !sum_y_.allocate(segments, "allocating bed y sums") ||
        !sum_x_.allocate(segments, "allocating bed x sums") ||
        !offsets_.allocate(segments + 1, "allocating bed sort offsets")) {
      return false;
    }
    workspace_.keys[0] = keys_[0].get();
    workspace_.keys[1] = keys_[1].get();
    workspace_.points[0] = points_[0].get();
    workspace_.points[1] = points_[1].get();
    workspace_.row_to_record = row_to_record_.get();
    workspace_.active_count = active_count_.get();
    workspace_.crop = crop_.get();
    workspace_.sum_y = sum_y_.get();
    workspace_.sum_x = sum_x_.get();
    workspace_.offsets = offsets_.get();
    std::array<std::int32_t, seeon::trt::kBedFinalizeSegments + 1> offsets{};
    for (int row = 0; row <= seeon::trt::kBedFinalizeSegments; ++row)
      offsets[static_cast<std::size_t>(row)] = row * seeon::trt::kBedFinalizePixels;
    if (!cuda_ok(cudaMemcpy(workspace_.offsets, offsets.data(), sizeof(offsets),
                            cudaMemcpyHostToDevice),
                 "initializing bed sort offsets")) {
      return false;
    }
    std::string error;
    if (!seeon::trt::query_bed_finalize_workspace_temp_bytes(&workspace_, &error)) {
      std::fprintf(stderr, "postprocess_gpu_test: CUB workspace query failed: %s\n", error.c_str());
      return false;
    }
    if (!temp_.allocate(workspace_.cub_temp_bytes, "allocating CUB temporary storage")) return false;
    workspace_.cub_temp = temp_.get();
    return true;
  }

  [[nodiscard]] BedFinalizeWorkspace* get() { return &workspace_; }
  [[nodiscard]] std::size_t bytes() const {
    constexpr std::size_t entries = seeon::trt::kBedFinalizeEntries;
    constexpr std::size_t segments = seeon::trt::kBedFinalizeSegments;
    return 2 * entries * (sizeof(std::uint64_t) + sizeof(std::uint32_t)) +
           segments * (sizeof(std::int32_t) * 2 + sizeof(int4) + sizeof(std::int64_t) * 2) +
           (segments + 1) * sizeof(std::int32_t) + workspace_.cub_temp_bytes;
  }
  [[nodiscard]] std::size_t temp_bytes() const { return workspace_.cub_temp_bytes; }

 private:
  DeviceAllocation<std::uint64_t> keys_[2];
  DeviceAllocation<std::uint32_t> points_[2];
  DeviceAllocation<std::int32_t> row_to_record_;
  DeviceAllocation<std::int32_t> active_count_;
  DeviceAllocation<int4> crop_;
  DeviceAllocation<std::int64_t> sum_y_;
  DeviceAllocation<std::int64_t> sum_x_;
  DeviceAllocation<std::int32_t> offsets_;
  DeviceAllocation<std::byte> temp_;
  BedFinalizeWorkspace workspace_{};
};

struct Fixture {
  std::vector<float> rows;
  std::vector<float> prototypes;
  AffineMetadata affine;
  const char* label;
};

void set_channel(std::vector<float>* prototypes, int channel,
                 const std::vector<std::pair<int, int>>& points) {
  constexpr int pixels = seeon::trt::kBedFinalizePixels;
  float* destination = prototypes->data() + static_cast<std::size_t>(channel) * pixels;
  std::fill(destination, destination + pixels, -1.0F);
  for (const auto& [y, x] : points) destination[y * 160 + x] = 1.0F;
}

std::vector<std::pair<int, int>> mask_points(int kind) {
  std::vector<std::pair<int, int>> points;
  const int count = kind == 0 ? 0 : kind == 1 ? 1 : kind == 2 ? 47 : kind == 3 ? 48 :
                    kind == 4 ? 49 : kind == 5 ? 25600 : kind == 6 ? 17 :
                    kind == 7 ? 64 : kind == 8 ? 936 : 0;
  points.reserve(count);
  if (kind == 7) {  // Deliberately contains tied rays from its centroid.
    for (int radius = 1; radius <= 16; ++radius) {
      points.emplace_back(80, 80 + radius);
      points.emplace_back(80, 80 - radius);
      points.emplace_back(80 + radius, 80);
      points.emplace_back(80 - radius, 80);
    }
    return points;
  }
  for (int index = 0; index < count; ++index) {
    const int pixel = kind == 6 ? ((index * 941) % 25600) : index;
    points.emplace_back(pixel / 160, pixel % 160);
  }
  return points;
}

Fixture make_fixture(int seed) {
  Fixture fixture{
      std::vector<float>(seeon::trt::kPostprocessTensorRows * seeon::trt::kPostprocessBedRowStride,
                         0.0F),
      std::vector<float>(32 * seeon::trt::kBedFinalizePixels, -1.0F),
      seeon::perception::letterbox_affine(719 + seed * 11, 1281 - seed * 17),
      seed == 0 ? "seed-0" : seed == 1 ? "seed-1" : "seed-2"};
  for (int kind = 0; kind < 9; ++kind) set_channel(&fixture.prototypes, kind, mask_points(kind));
  std::fill(fixture.prototypes.begin() + 9 * seeon::trt::kBedFinalizePixels,
            fixture.prototypes.begin() + 10 * seeon::trt::kBedFinalizePixels, 0.0F);
  for (int row = 0; row < seeon::trt::kPostprocessTensorRows; ++row) {
    float* value = fixture.rows.data() + row * seeon::trt::kPostprocessBedRowStride;
    const int boundary = row % 50;  // 50 score/class/crop/inverse-geometry boundaries per seed.
    const int mask = (boundary + seed) % 11;
    value[0] = boundary == 0 ? -0.01F : 3.999F + static_cast<float>((boundary * 37 + seed) % 620);
    value[1] = boundary == 1 ? -0.01F : 2.999F + static_cast<float>((boundary * 19 + seed) % 620);
    value[2] = boundary == 2 ? 640.01F : value[0] + 1.0F + static_cast<float>(boundary % 31);
    value[3] = boundary == 3 ? 640.01F : value[1] + 1.0F + static_cast<float>((boundary * 3) % 29);
    value[4] = boundary == 4 ? std::nextafter(0.25F, 0.0F)
               : boundary == 5 ? 0.25F
               : boundary == 6 ? std::nextafter(0.25F, 1.0F)
               : 0.30F + static_cast<float>((row + seed) % 11) * 0.01F;
    value[5] = boundary == 7 ? 58.999F : boundary == 8 ? 59.0F : boundary == 9 ? 59.999F : 59.0F;
    value[6 + mask] = 1.0F;  // Includes negative, zero, sparse, tied-ray, and trial-936 masks.
    // The final 50 are deterministic class rejects; score/class boundaries above add rejects.
    if (row >= 250) {
      value[5] = (row & 1) == 0 ? 0.0F : 58.0F;
      value[4] = 0.9F;
    }
  }
  // Explicit active-count boundaries and source-order records.
  for (int index = 0; index < 11; ++index) {
    float* value = fixture.rows.data() + index * seeon::trt::kPostprocessBedRowStride;
    if (index < 6) {
      std::fill(value + 6, value + seeon::trt::kPostprocessBedRowStride, 0.0F);
      value[6 + index] = 1.0F;  // 0, 1, 47, 48, 49, and 25,600 active pixels.
    }
    value[0] = 0.0F;
    value[1] = 0.0F;
    value[2] = static_cast<float>(fixture.affine.tensor_width);
    value[3] = static_cast<float>(fixture.affine.tensor_height);
  }
  fixture.rows[7 * seeon::trt::kPostprocessBedRowStride + 5] = 59.0F;
  return fixture;
}

BedFinalizeGeometry geometry_for(const AffineMetadata& affine) {
  return {affine.source_height, affine.source_width, affine.tensor_height, affine.tensor_width,
          affine.gain, affine.box_pad_x, affine.box_pad_y, affine.keypoint_pad_x,
          affine.keypoint_pad_y};
}

bool check_records(const std::vector<ParsedBedRegion>& expected,
                   std::span<const PackedBedRecord> actual, const char* label) {
  if (expected.size() != actual.size()) {
    std::fprintf(stderr, "postprocess_gpu_test: %s record count expected=%zu actual=%zu\n", label,
                 expected.size(), actual.size());
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& region = expected[index];
    const PackedBedRecord& record = actual[index];
    const std::array<int, 4> box = {region.bounds.x1, region.bounds.y1, region.bounds.x2,
                                    region.bounds.y2};
    if (!std::equal(box.begin(), box.end(), record.box) ||
        std::abs(record.confidence - region.bounds.confidence) > 6e-6 ||
        record.point_count != static_cast<int>(region.polygon.size()) || record.pad != 0) {
      std::fprintf(stderr, "postprocess_gpu_test: %s serialized record mismatch index=%zu\n", label,
                   index);
      return false;
    }
    for (int point = 0; point < seeon::trt::kPostprocessBedMaxPoints; ++point) {
      const bool used = point < record.point_count;
      const int x = used ? region.polygon[static_cast<std::size_t>(point)].first : 0;
      const int y = used ? region.polygon[static_cast<std::size_t>(point)].second : 0;
      if (record.points[point][0] != x || record.points[point][1] != y) {
        std::fprintf(stderr, "postprocess_gpu_test: %s point/padding mismatch record=%zu point=%d\n",
                     label, index, point);
        return false;
      }
    }
  }
  return true;
}

struct Run {
  std::vector<std::byte> serialized;
  float milliseconds = 0.0F;
};

bool run_fixture(const Fixture& fixture, BedWorkspace* workspace, Run* run) {
  DeviceAllocation<float> rows;
  DeviceAllocation<float> prototypes;
  DeviceAllocation<PackedBedRecord> records;
  DeviceAllocation<PostprocessChannelHeader> header;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t finish = nullptr;
  const std::size_t row_bytes = fixture.rows.size() * sizeof(float);
  const std::size_t prototype_bytes = fixture.prototypes.size() * sizeof(float);
  bool ok = rows.allocate(fixture.rows.size(), "allocating bed rows") &&
            prototypes.allocate(fixture.prototypes.size(), "allocating bed prototypes") &&
            records.allocate(seeon::trt::kPostprocessTensorRows, "allocating packed bed records") &&
            header.allocate(1, "allocating bed header") &&
            cuda_ok(cudaStreamCreate(&stream), "creating bed stream") &&
            cuda_ok(cudaEventCreate(&start), "creating bed start event") &&
            cuda_ok(cudaEventCreate(&finish), "creating bed finish event") &&
            cuda_ok(cudaMemcpyAsync(rows.get(), fixture.rows.data(), row_bytes, cudaMemcpyHostToDevice,
                                    stream), "copying synthetic bed rows") &&
            cuda_ok(cudaMemcpyAsync(prototypes.get(), fixture.prototypes.data(), prototype_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "copying synthetic prototypes") &&
            cuda_ok(cudaEventRecord(start, stream), "recording bed start");
  std::string error;
  if (ok && !seeon::trt::finalize_bed_rows_device(rows.get(), prototypes.get(), records.get(),
                                                   header.get(), workspace->get(),
                                                   geometry_for(fixture.affine), stream, &error)) {
    std::fprintf(stderr, "postprocess_gpu_test: production finalizer failed: %s\n", error.c_str());
    ok = false;
  }
  PostprocessChannelHeader result_header{};
  if (ok) {
    ok = cuda_ok(cudaEventRecord(finish, stream), "recording bed finish") &&
         cuda_ok(cudaMemcpyAsync(&result_header, header.get(), sizeof(result_header),
                                 cudaMemcpyDeviceToHost, stream),
                 "copying bed header") &&
         cuda_ok(cudaStreamSynchronize(stream), "synchronizing bed finalizer") &&
         cuda_ok(cudaEventElapsedTime(&run->milliseconds, start, finish), "measuring bed finalizer");
  }
  if (ok && (result_header.kernel_executed != 1 || result_header.count < 0 ||
             result_header.count > seeon::trt::kPostprocessTensorRows)) {
    std::fprintf(stderr, "postprocess_gpu_test: invalid bed header executed=%d count=%d\n",
                 result_header.kernel_executed, result_header.count);
    ok = false;
  }
  if (ok) {
    const std::size_t record_bytes = static_cast<std::size_t>(result_header.count) * sizeof(PackedBedRecord);
    run->serialized.resize(sizeof(result_header) + record_bytes);
    std::memcpy(run->serialized.data(), &result_header, sizeof(result_header));
    if (record_bytes != 0) {
      ok = cuda_ok(cudaMemcpyAsync(run->serialized.data() + sizeof(result_header), records.get(),
                                   record_bytes, cudaMemcpyDeviceToHost, stream),
                   "copying packed bed records") &&
           cuda_ok(cudaStreamSynchronize(stream), "synchronizing packed bed records");
    }
  }
  if (finish != nullptr) cudaEventDestroy(finish);
  if (start != nullptr) cudaEventDestroy(start);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return ok;
}

bool verify_fixture(const Fixture& fixture, BedWorkspace* workspace, Run* first) {
  if (!run_fixture(fixture, workspace, first)) return false;
  const auto expected = seeon::perception::parse_bed_rows(
      std::span<const float>{fixture.rows}, std::span<const float>{fixture.prototypes},
      fixture.affine, 0.25, 48);
  const auto* header = reinterpret_cast<const PostprocessChannelHeader*>(first->serialized.data());
  const auto* records = reinterpret_cast<const PackedBedRecord*>(
      first->serialized.data() + sizeof(PostprocessChannelHeader));
  return header->kernel_executed == 1 && header->count == static_cast<int>(expected.size()) &&
         check_records(expected, {records, expected.size()}, fixture.label);
}

float percentile95(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[(samples.size() - 1) * 95 / 100];
}

}  // namespace

int main() {
  if (!verify_header_validation() || !verify_invalid_headers_skip_metadata_conversion())
    return EXIT_FAILURE;
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "enumerating CUDA devices") || device_count < 1 ||
      !cuda_ok(cudaSetDevice(0), "selecting CUDA device")) return EXIT_FAILURE;
  BedWorkspace workspace;
  if (!workspace.allocate()) return EXIT_FAILURE;

  std::vector<Run> runs(3);
  for (int seed = 0; seed < 3; ++seed) {
    const Fixture fixture = make_fixture(seed);
    if (!verify_fixture(fixture, &workspace, &runs[static_cast<std::size_t>(seed)])) return EXIT_FAILURE;
  }
  const Fixture worst = [] {
    Fixture fixture = make_fixture(0);
    fixture.label = "worst-300-accepted-empty-mask";
    for (int row = 0; row < seeon::trt::kPostprocessTensorRows; ++row) {
      float* value = fixture.rows.data() + row * seeon::trt::kPostprocessBedRowStride;
      value[4] = 0.25F;
      value[5] = 59.0F;
      std::fill(value + 6, value + seeon::trt::kPostprocessBedRowStride, 0.0F);
      value[6] = 1.0F;
    }
    return fixture;
  }();
  Run worst_run;
  if (!verify_fixture(worst, &workspace, &worst_run)) return EXIT_FAILURE;
  Run rerun;
  if (!run_fixture(make_fixture(0), &workspace, &rerun) ||
      rerun.serialized != runs[0].serialized) {
    std::fprintf(stderr, "postprocess_gpu_test: deterministic serialized rerun mismatch\n");
    return EXIT_FAILURE;
  }

  std::vector<float> milliseconds;
  for (const Run& run : runs) milliseconds.push_back(run.milliseconds);
  milliseconds.push_back(worst_run.milliseconds);
  milliseconds.push_back(rerun.milliseconds);
  const std::size_t max_serialized = std::max(
      {runs[0].serialized.size(), runs[1].serialized.size(), runs[2].serialized.size(),
       worst_run.serialized.size(), rerun.serialized.size()});
  constexpr std::size_t bed_bytes = seeon::trt::postprocess_bed_transfer_bytes(300);
  std::size_t max_person_transfer_bytes = 0;
  std::size_t pose_only_transfer_bytes = 0;
  if (bed_bytes != 124808 ||
      !seeon::trt::postprocess_transfer_bytes(300, 300, 300, &max_person_transfer_bytes) ||
      max_person_transfer_bytes != 200424 ||
      !seeon::trt::postprocess_transfer_bytes(300, 0, 300, &pose_only_transfer_bytes) ||
      pose_only_transfer_bytes != 193224 ||
      seeon::trt::postprocess_transfer_bytes(301, 0, 0, &pose_only_transfer_bytes) ||
      seeon::trt::postprocess_pose_transfer_bytes(300) +
              seeon::trt::postprocess_person_transfer_bytes(300) + bed_bytes !=
          200424 ||
      seeon::trt::postprocess_pose_transfer_bytes(300) +
              seeon::trt::postprocess_person_transfer_bytes(0) + bed_bytes !=
          193224) {
    std::fprintf(stderr, "postprocess_gpu_test: transfer formula mismatch\n");
    return EXIT_FAILURE;
  }
  std::printf(
      "POSTPROCESS_GPU_RECEIPT fixtures=150 seeds=3 rows_per_seed=300 accepted_per_seed=241 "
      "rejected_per_seed=59 worst_empty_mask_records=300 parity_mismatches=0 "
      "masks=empty,full,sparse,tied-ray,trial936 dot_thresholds=negative,zero "
      "deterministic_rerun_bytes=1 headers_executed=1 max_serialized_bytes=%zu "
      "gpu_ms_p95_per_call=%.3f gpu_ms_max_per_call=%.3f workspace_bytes=%zu cub_temp_bytes=%zu "
      "bed_d2h_max_bytes=%zu person_total_d2h_bytes=%zu pose_only_d2h_bytes=%zu "
      "raw_bed_row_d2h_bytes=0 raw_bed_prototype_d2h_bytes=0\n",
      max_serialized, percentile95(milliseconds), *std::max_element(milliseconds.begin(), milliseconds.end()),
      workspace.bytes(), workspace.temp_bytes(), bed_bytes, seeon::trt::kPostprocessMaxPersonTransferBytes,
      seeon::trt::kPostprocessPoseOnlyTransferBytes);
  return EXIT_SUCCESS;
}
