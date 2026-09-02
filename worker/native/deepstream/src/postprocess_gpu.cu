#include "postprocess_gpu.hpp"
#include "pinned_host_atan2.cuh"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace seeon::trt {
namespace {

constexpr int kMaskWidth = 160;
constexpr int kMaskHeight = 160;
constexpr int kMaskChannels = 32;
constexpr std::uint64_t kInactiveKey = UINT64_MAX;
constexpr std::uint64_t kCanonicalNaNBits = 0x7ff8000000000000ULL;

static_assert(kBedFinalizeSegments == kPostprocessTensorRows);
static_assert(kBedFinalizePixels == kMaskWidth * kMaskHeight);

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
    for (int column = 0; column < kStride; ++column)
      compact_words[compact_offset + column] = source_words[source_offset + column];
    ++count;
  }
  header->kernel_executed = 1;
  header->count = count;
}

__device__ int inverse_box_coordinate(double value, int pad, double gain, int limit) {
  const double scaled = (value - static_cast<double>(pad)) / gain;
  return static_cast<int>(scaled < 0.0 ? 0.0 : (scaled > limit ? limit : scaled));
}

__device__ int inverse_keypoint_coordinate(double value, double pad, double gain, int limit) {
  const double scaled = (value - pad) / gain;
  return static_cast<int>(scaled < 0.0 ? 0.0 : (scaled > limit ? limit : scaled));
}

// One lane deliberately establishes source-order record assignment before the
// massively parallel phases consume it.
__global__ void bed_phase1_rows_kernel(const float* rows, PackedBedRecord* records,
                                       PostprocessChannelHeader* header,
                                       BedFinalizeWorkspace workspace,
                                       BedFinalizeGeometry geometry) {
  int record_count = 0;
  const double width_ratio = static_cast<double>(kMaskWidth) / geometry.tensor_width;
  const double height_ratio = static_cast<double>(kMaskHeight) / geometry.tensor_height;
  for (int row_index = 0; row_index < kPostprocessTensorRows; ++row_index) {
    workspace.row_to_record[row_index] = -1;
    workspace.active_count[row_index] = 0;
    workspace.sum_y[row_index] = 0;
    workspace.sum_x[row_index] = 0;
    const float* row = rows + row_index * kPostprocessBedRowStride;
    const float score = row[4];
    const bool accepted = static_cast<int>(static_cast<double>(row[5])) == 59 &&
                          !(score < 0.25F);
    if (!accepted) continue;

    workspace.row_to_record[row_index] = record_count;
    PackedBedRecord& record = records[record_count++];
    record.box[0] = inverse_box_coordinate(static_cast<double>(row[0]), geometry.box_pad_x,
                                           geometry.gain, geometry.source_width);
    record.box[1] = inverse_box_coordinate(static_cast<double>(row[1]), geometry.box_pad_y,
                                           geometry.gain, geometry.source_height);
    record.box[2] = inverse_box_coordinate(static_cast<double>(row[2]), geometry.box_pad_x,
                                           geometry.gain, geometry.source_width);
    record.box[3] = inverse_box_coordinate(static_cast<double>(row[3]), geometry.box_pad_y,
                                           geometry.gain, geometry.source_height);
    record.confidence = static_cast<double>(score);
    record.point_count = 0;
    record.pad = 0;
    for (int point = 0; point < kPostprocessBedMaxPoints; ++point) {
      record.points[point][0] = 0;
      record.points[point][1] = 0;
    }

    const int raw_left = static_cast<int>(floor(static_cast<double>(row[0]) * width_ratio));
    const int raw_top = static_cast<int>(floor(static_cast<double>(row[1]) * height_ratio));
    const int raw_right = static_cast<int>(ceil(static_cast<double>(row[2]) * width_ratio));
    const int raw_bottom = static_cast<int>(ceil(static_cast<double>(row[3]) * height_ratio));
    workspace.crop[row_index] = make_int4(raw_left < 0 ? 0 : raw_left,
                                          raw_top < 0 ? 0 : raw_top,
                                          raw_right > kMaskWidth ? kMaskWidth : raw_right,
                                          raw_bottom > kMaskHeight ? kMaskHeight : raw_bottom);
  }
  header->kernel_executed = 1;
  header->count = record_count;
}

__global__ void bed_phase2_mask_kernel(const float* rows, const float* prototypes,
                                       BedFinalizeWorkspace workspace) {
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= kBedFinalizeEntries) return;
  const int row_index = entry / kBedFinalizePixels;
  const std::uint32_t point = static_cast<std::uint32_t>(entry % kBedFinalizePixels);
  workspace.points[0][entry] = point;
  workspace.keys[0][entry] = kInactiveKey;
  if (workspace.row_to_record[row_index] < 0) return;

  const int y = point / kMaskWidth;
  const int x = point % kMaskWidth;
  const int4 crop = workspace.crop[row_index];
  if (x < crop.x || x >= crop.z || y < crop.y || y >= crop.w) return;

  const float* row = rows + row_index * kPostprocessBedRowStride;
  const int prototype_offset = y * kMaskWidth + x;
  double value = 0.0;
#pragma unroll
  for (int channel = 0; channel < kMaskChannels; ++channel) {
    const double term = __dmul_rn(static_cast<double>(row[6 + channel]),
                                  static_cast<double>(prototypes[channel * kBedFinalizePixels +
                                                                  prototype_offset]));
    value = __dadd_rn(value, term);
  }
  if (!(value > 0.0)) return;
  atomicAdd(&workspace.active_count[row_index], 1);
  atomicAdd(reinterpret_cast<unsigned long long*>(&workspace.sum_y[row_index]),
            static_cast<unsigned long long>(y));
  atomicAdd(reinterpret_cast<unsigned long long*>(&workspace.sum_x[row_index]),
            static_cast<unsigned long long>(x));
  // This temporary marker is replaced by the angle key after all exact sums are complete.
  workspace.keys[0][entry] = 0;
}

__global__ void bed_phase3_angle_kernel(PostprocessChannelHeader* header,
                                        BedFinalizeWorkspace workspace) {
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= kBedFinalizeEntries || workspace.keys[0][entry] == kInactiveKey) return;
  const int row_index = entry / kBedFinalizePixels;
  const int count = workspace.active_count[row_index];
  if (count <= 0) {
    workspace.keys[0][entry] = kInactiveKey;
    return;
  }
  const std::uint32_t point = workspace.points[0][entry];
  const int y = point / kMaskWidth;
  const int x = point % kMaskWidth;
  const double center_y = static_cast<double>(workspace.sum_y[row_index]) / count;
  const double center_x = static_cast<double>(workspace.sum_x[row_index]) / count;
  const double angle = pinned_host_atan2(static_cast<double>(y) - center_y,
                                         static_cast<double>(x) - center_x);
  const std::uint64_t bits = static_cast<std::uint64_t>(__double_as_longlong(angle));
  if (bits == kCanonicalNaNBits || !isfinite(angle)) {
    workspace.keys[0][entry] = kInactiveKey;
    atomicExch(&header->kernel_executed, 0);
    return;
  }
  workspace.keys[0][entry] = (bits & (UINT64_C(1) << 63)) ? ~bits : bits ^ (UINT64_C(1) << 63);
}

__global__ void bed_phase4_records_kernel(PackedBedRecord* records, BedFinalizeWorkspace workspace,
                                          BedFinalizeGeometry geometry,
                                          const std::uint32_t* sorted_points) {
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= kPostprocessTensorRows) return;
  const int record_index = workspace.row_to_record[row_index];
  if (record_index < 0) return;
  PackedBedRecord& record = records[record_index];
  const int count = workspace.active_count[row_index];
  record.point_count = count > kPostprocessBedMaxPoints ? kPostprocessBedMaxPoints : count;
  if (count == 0) return;
  const double width_ratio = static_cast<double>(kMaskWidth) / geometry.tensor_width;
  const double height_ratio = static_cast<double>(kMaskHeight) / geometry.tensor_height;
  const int offset = row_index * kBedFinalizePixels;
  for (int sample = 0; sample < record.point_count; ++sample) {
    const int rank = count > kPostprocessBedMaxPoints
                         ? static_cast<int>(static_cast<double>(count - 1) * sample /
                                            (kPostprocessBedMaxPoints - 1))
                         : sample;
    const std::uint32_t point = sorted_points[offset + rank];
    const int y = point / kMaskWidth;
    const int x = point % kMaskWidth;
    record.points[sample][0] = inverse_keypoint_coordinate(static_cast<double>(x) / width_ratio,
                                                            geometry.keypoint_pad_x, geometry.gain,
                                                            geometry.source_width);
    record.points[sample][1] = inverse_keypoint_coordinate(static_cast<double>(y) / height_ratio,
                                                            geometry.keypoint_pad_y, geometry.gain,
                                                            geometry.source_height);
  }
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

bool valid_bed_workspace(const BedFinalizeWorkspace& workspace) {
  return workspace.keys[0] != nullptr && workspace.keys[1] != nullptr &&
         workspace.points[0] != nullptr && workspace.points[1] != nullptr &&
         workspace.row_to_record != nullptr && workspace.active_count != nullptr &&
         workspace.crop != nullptr && workspace.sum_y != nullptr && workspace.sum_x != nullptr &&
         workspace.offsets != nullptr && workspace.cub_temp != nullptr && workspace.cub_temp_bytes != 0;
}

bool valid_required_channel_header(const PostprocessChannelHeader& header) {
  return header.kernel_executed == 1 && header.count >= 0 &&
         header.count <= kPostprocessTensorRows;
}

}  // namespace

bool validate_postprocess_channel_headers(const PostprocessChannelHeader& pose,
                                          const PostprocessChannelHeader& person,
                                          const PostprocessChannelHeader& bed,
                                          bool person_required, std::string* error) {
  const bool person_valid = person_required
                                ? valid_required_channel_header(person)
                                : person.kernel_executed == 0 && person.count == 0;
  if (valid_required_channel_header(pose) && valid_required_channel_header(bed) && person_valid)
    return true;
  if (error != nullptr) *error = "postprocess_header_invalid";
  return false;
}

bool postprocess_transfer_bytes(std::size_t pose_count, std::size_t person_count,
                                std::size_t bed_count, std::size_t* transfer_bytes) {
  if (transfer_bytes == nullptr || pose_count > kPostprocessTensorRows ||
      person_count > kPostprocessTensorRows || bed_count > kPostprocessTensorRows) {
    return false;
  }
  constexpr std::size_t header_bytes = 3 * sizeof(PostprocessChannelHeader);
  constexpr std::size_t max_row_bytes =
      kPostprocessPoseRecordBytes + kPostprocessPersonRecordBytes + sizeof(PackedBedRecord);
  static_assert(kPostprocessTensorRows <=
                (std::numeric_limits<std::size_t>::max() - header_bytes) / max_row_bytes);
  *transfer_bytes = header_bytes + pose_count * kPostprocessPoseRecordBytes +
                    person_count * kPostprocessPersonRecordBytes +
                    bed_count * sizeof(PackedBedRecord);
  return true;
}

bool compact_pose_rows_device(const float* source_rows, float* compact_rows,
                              PostprocessChannelHeader* header, cudaStream_t stream,
                              std::string* error) {
  return launch_compaction<kPostprocessPoseRowStride, true>(source_rows, compact_rows, header,
                                                             stream, error);
}

bool compact_person_rows_device(const float* source_rows, float* compact_rows,
                                PostprocessChannelHeader* header, cudaStream_t stream,
                                std::string* error) {
  return launch_compaction<kPostprocessPersonRowStride, false>(source_rows, compact_rows, header,
                                                                stream, error);
}

bool query_bed_finalize_workspace_temp_bytes(BedFinalizeWorkspace* workspace,
                                             std::string* error) {
  if (workspace == nullptr || workspace->keys[0] == nullptr || workspace->keys[1] == nullptr ||
      workspace->points[0] == nullptr || workspace->points[1] == nullptr ||
      workspace->offsets == nullptr) {
    *error = "bed_postprocess_workspace_invalid";
    return false;
  }
  cub::DoubleBuffer<std::uint64_t> keys(workspace->keys[0], workspace->keys[1]);
  cub::DoubleBuffer<std::uint32_t> points(workspace->points[0], workspace->points[1]);
  std::size_t bytes = 0;
  const cudaError_t status = cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, bytes, keys, points, kBedFinalizeEntries, kBedFinalizeSegments, workspace->offsets,
      workspace->offsets + 1, 0, 64);
  if (status != cudaSuccess || bytes == 0) {
    *error = "bed_postprocess_cub_query_failed";
    return false;
  }
  workspace->cub_temp_bytes = bytes;
  return true;
}

bool finalize_bed_rows_device(const float* source_rows, const float* prototypes,
                              PackedBedRecord* records, PostprocessChannelHeader* header,
                              BedFinalizeWorkspace* workspace,
                              const BedFinalizeGeometry& geometry, cudaStream_t stream,
                              std::string* error) {
  if (source_rows == nullptr || prototypes == nullptr || records == nullptr || header == nullptr ||
      workspace == nullptr || !valid_bed_workspace(*workspace)) {
    *error = "bed_postprocess_destination_invalid";
    return false;
  }
  if (stream == nullptr || geometry.source_height <= 0 || geometry.source_width <= 0 ||
      geometry.tensor_height <= 0 || geometry.tensor_width <= 0 || geometry.gain <= 0.0) {
    *error = "bed_postprocess_arguments_invalid";
    return false;
  }
  (void)cudaGetLastError();
  bed_phase1_rows_kernel<<<1, 1, 0, stream>>>(source_rows, records, header, *workspace, geometry);
  if (cudaGetLastError() != cudaSuccess) {
    *error = "bed_postprocess_kernel_launch_failed";
    return false;
  }
  bed_phase2_mask_kernel<<<(kBedFinalizeEntries + 255) / 256, 256, 0, stream>>>(
      source_rows, prototypes, *workspace);
  if (cudaGetLastError() != cudaSuccess) {
    *error = "bed_postprocess_kernel_launch_failed";
    return false;
  }
  bed_phase3_angle_kernel<<<(kBedFinalizeEntries + 255) / 256, 256, 0, stream>>>(header, *workspace);
  if (cudaGetLastError() != cudaSuccess) {
    *error = "bed_postprocess_kernel_launch_failed";
    return false;
  }
  cub::DoubleBuffer<std::uint64_t> keys(workspace->keys[0], workspace->keys[1]);
  cub::DoubleBuffer<std::uint32_t> points(workspace->points[0], workspace->points[1]);
  const cudaError_t sort_status = cub::DeviceSegmentedRadixSort::SortPairs(
      workspace->cub_temp, workspace->cub_temp_bytes, keys, points, kBedFinalizeEntries,
      kBedFinalizeSegments, workspace->offsets, workspace->offsets + 1, 0, 64, stream);
  if (sort_status != cudaSuccess) {
    *error = "bed_postprocess_kernel_launch_failed";
    return false;
  }
  bed_phase4_records_kernel<<<(kPostprocessTensorRows + 255) / 256, 256, 0, stream>>>(
      records, *workspace, geometry, points.Current());
  if (cudaGetLastError() != cudaSuccess) {
    *error = "bed_postprocess_kernel_launch_failed";
    return false;
  }
  return true;
}

}  // namespace seeon::trt
