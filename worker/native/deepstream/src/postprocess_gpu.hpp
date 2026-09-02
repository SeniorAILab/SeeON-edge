#pragma once

#include <cstddef>
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
inline constexpr int kPostprocessBedRowStride = 38;
inline constexpr int kPostprocessBedMaxPoints = 48;
inline constexpr int kBedFinalizeSegments = 300;
inline constexpr int kBedFinalizePixels = 160 * 160;
inline constexpr int kBedFinalizeEntries = kBedFinalizeSegments * kBedFinalizePixels;
static_assert(kBedFinalizeEntries == 7680000);

// This is the sole host-visible representation of a finalized bed region.
// Its ABI is intentionally independent of PerceptionFrameV1.
struct PackedBedRecord {
  std::int32_t box[4];
  double confidence;
  std::int32_t point_count;
  std::int32_t pad;
  std::int32_t points[kPostprocessBedMaxPoints][2];
};
static_assert(sizeof(PackedBedRecord) == 416);
static_assert(offsetof(PackedBedRecord, box) == 0);
static_assert(offsetof(PackedBedRecord, confidence) == 16);
static_assert(offsetof(PackedBedRecord, point_count) == 24);
static_assert(offsetof(PackedBedRecord, pad) == 28);
static_assert(offsetof(PackedBedRecord, points) == 32);

inline constexpr std::size_t kPostprocessPoseRecordBytes =
    kPostprocessPoseRowStride * sizeof(float);
inline constexpr std::size_t kPostprocessPersonRecordBytes =
    kPostprocessPersonRowStride * sizeof(float);
inline constexpr std::size_t kPostprocessBedChannelMaxBytes =
    sizeof(PostprocessChannelHeader) + kPostprocessTensorRows * sizeof(PackedBedRecord);
inline constexpr std::size_t postprocess_pose_transfer_bytes(std::size_t count) {
  return sizeof(PostprocessChannelHeader) + count * kPostprocessPoseRecordBytes;
}
inline constexpr std::size_t postprocess_person_transfer_bytes(std::size_t count) {
  return sizeof(PostprocessChannelHeader) + count * kPostprocessPersonRecordBytes;
}
inline constexpr std::size_t postprocess_bed_transfer_bytes(std::size_t count) {
  return sizeof(PostprocessChannelHeader) + count * sizeof(PackedBedRecord);
}
inline constexpr std::size_t kPostprocessMaxPersonTransferBytes =
    postprocess_pose_transfer_bytes(kPostprocessTensorRows) +
    postprocess_person_transfer_bytes(kPostprocessTensorRows) +
    postprocess_bed_transfer_bytes(kPostprocessTensorRows);
inline constexpr std::size_t kPostprocessPoseOnlyTransferBytes =
    postprocess_pose_transfer_bytes(kPostprocessTensorRows) +
    postprocess_person_transfer_bytes(0) +
    postprocess_bed_transfer_bytes(kPostprocessTensorRows);
static_assert(kPostprocessBedChannelMaxBytes == 124808);
static_assert(kPostprocessMaxPersonTransferBytes == 200424);
static_assert(kPostprocessPoseOnlyTransferBytes == 193224);

// Validates the headers copied back from the device before their counts are
// used to schedule row transfers or convert inference metadata.
[[nodiscard]] bool validate_postprocess_channel_headers(
    const PostprocessChannelHeader& pose, const PostprocessChannelHeader& person,
    const PostprocessChannelHeader& bed, bool person_required, std::string* error);

// Calculates the aggregate device-to-host postprocess transfer size only for
// validated channel counts.
[[nodiscard]] bool postprocess_transfer_bytes(
    std::size_t pose_count, std::size_t person_count, std::size_t bed_count,
    std::size_t* transfer_bytes);

struct BedFinalizeGeometry {
  int source_height;
  int source_width;
  int tensor_height;
  int tensor_width;
  double gain;
  int box_pad_x;
  int box_pad_y;
  double keypoint_pad_x;
  double keypoint_pad_y;
};

// All device allocations in this workspace are owned by one Trt Workspace and
// are initialized at load, never while processing a frame.
struct BedFinalizeWorkspace {
  std::uint64_t* keys[2]{};
  std::uint32_t* points[2]{};
  std::int32_t* row_to_record{};
  std::int32_t* active_count{};
  int4* crop{};
  std::int64_t* sum_y{};
  std::int64_t* sum_x{};
  std::int32_t* offsets{};
  void* cub_temp{};
  std::size_t cub_temp_bytes{};
};

// Queries CUB only after keys, points, and offsets have been allocated and
// offsets contains the fixed 300 x 25,600 segmented layout.
[[nodiscard]] bool query_bed_finalize_workspace_temp_bytes(
    BedFinalizeWorkspace* workspace, std::string* error);

// Enqueues source-order filtering and full-row byte copies. The caller owns all
// buffers through stream completion.
[[nodiscard]] bool compact_pose_rows_device(
    const float* source_rows, float* compact_rows, PostprocessChannelHeader* header,
    cudaStream_t stream, std::string* error);
[[nodiscard]] bool compact_person_rows_device(
    const float* source_rows, float* compact_rows, PostprocessChannelHeader* header,
    cudaStream_t stream, std::string* error);
[[nodiscard]] bool finalize_bed_rows_device(
    const float* source_rows, const float* prototypes, PackedBedRecord* records,
    PostprocessChannelHeader* header, BedFinalizeWorkspace* workspace,
    const BedFinalizeGeometry& geometry, cudaStream_t stream, std::string* error);

}  // namespace seeon::trt
