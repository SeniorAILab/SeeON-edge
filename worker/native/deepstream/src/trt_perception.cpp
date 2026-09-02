#include "trt_perception.hpp"

#include "copy_telemetry.hpp"
#include "postprocess_gpu.hpp"
#include "preprocess_gpu.hpp"
#include "source_runtime.hpp"
#include "workspace_pool.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

namespace seeon::trt {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "tensorrt: %s\n", message);
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

struct EngineSlot {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::string input_name;
  std::vector<std::string> output_names;
};

struct Workspace {
  std::unique_ptr<nvinfer1::IExecutionContext> pose_context;
  std::unique_ptr<nvinfer1::IExecutionContext> person_context;
  std::unique_ptr<nvinfer1::IExecutionContext> bed_context;
  cudaStream_t stream = nullptr;
  cudaEvent_t timing_start = nullptr;
  cudaEvent_t timing_end = nullptr;
  float* device_input = nullptr;
  float* device_pose = nullptr;
  float* device_person = nullptr;
  float* device_bed = nullptr;
  float* device_bed_prototypes = nullptr;
  float* device_pose_compact = nullptr;
  float* device_person_compact = nullptr;
  PackedBedRecord* device_bed_records = nullptr;
  BedFinalizeWorkspace bed_finalize;
  PostprocessChannelHeader* device_pose_header = nullptr;
  PostprocessChannelHeader* device_person_header = nullptr;
  PostprocessChannelHeader* device_bed_header = nullptr;
  std::vector<float> host_pose;
  std::vector<float> host_person;
  PostprocessChannelHeader host_pose_header{};
  PostprocessChannelHeader host_person_header{};
  PostprocessChannelHeader host_bed_header{};
  std::vector<PackedBedRecord> host_bed_records;

  ~Workspace() {
    if (timing_start != nullptr) cudaEventDestroy(timing_start);
    if (timing_end != nullptr) cudaEventDestroy(timing_end);
    for (float* pointer : {device_input, device_pose, device_person, device_bed,
                           device_bed_prototypes, device_pose_compact, device_person_compact}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    if (device_bed_records != nullptr) cudaFree(device_bed_records);
    for (std::uint64_t* pointer : {bed_finalize.keys[0], bed_finalize.keys[1]}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    for (std::uint32_t* pointer : {bed_finalize.points[0], bed_finalize.points[1]}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    for (void* pointer : {static_cast<void*>(bed_finalize.row_to_record),
                          static_cast<void*>(bed_finalize.active_count),
                          static_cast<void*>(bed_finalize.crop),
                          static_cast<void*>(bed_finalize.sum_y),
                          static_cast<void*>(bed_finalize.sum_x),
                          static_cast<void*>(bed_finalize.offsets),
                          bed_finalize.cub_temp}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    for (PostprocessChannelHeader* pointer :
         {device_pose_header, device_person_header, device_bed_header}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

bool read_file(const std::string& path, std::vector<char>* bytes) {
  std::ifstream input{path, std::ios::binary};
  if (!input) return false;
  bytes->assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return !bytes->empty();
}

constexpr int kMaxTensorSide = seeon::perception::kLetterboxSize;
constexpr std::size_t kInputCapacity = 3ULL * kMaxTensorSide * kMaxTensorSide;
constexpr std::size_t kPoseOutput = 300ULL * 57ULL;
constexpr std::size_t kPersonOutput = 300ULL * 6ULL;
constexpr std::size_t kBedOutput = 300ULL * 38ULL;
constexpr std::size_t kBedPrototypes = 32ULL * 160ULL * 160ULL;
constexpr std::size_t kInferenceWorkspaces = 4;

}  // namespace

class TrtPerception::Impl {
 public:
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  EngineSlot pose;
  EngineSlot person;
  EngineSlot bed;
  std::vector<std::unique_ptr<Workspace>> workspaces;
  BoundedPool<Workspace> pool;
  deepstream::CopyTelemetry copy_telemetry;

  bool load_engine(const std::string& path, EngineSlot* slot, std::string* error) {
    std::vector<char> bytes;
    if (!read_file(path, &bytes)) {
      *error = "engine_missing: " + path;
      return false;
    }
    slot->engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (slot->engine == nullptr) {
      *error = "engine_deserialize_failed: " + path;
      return false;
    }
    const int tensors = slot->engine->getNbIOTensors();
    for (int index = 0; index < tensors; ++index) {
      const char* name = slot->engine->getIOTensorName(index);
      if (slot->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        slot->input_name = name;
      } else {
        slot->output_names.emplace_back(name);
      }
    }
    std::sort(slot->output_names.begin(), slot->output_names.end());
    if (slot->input_name.empty() || slot->output_names.empty()) {
      *error = "engine_tensor_names_invalid: " + path;
      return false;
    }
    return true;
  }

  static bool enqueue_engine(const EngineSlot& slot, nvinfer1::IExecutionContext* context,
                             Workspace& workspace, int tensor_height, int tensor_width,
                             float* device_rows, std::size_t rows_capacity, float* host_rows,
                             float* device_extra, std::size_t extra_capacity,
                             float* host_extra, bool copy_rows, std::string* error) {
    const nvinfer1::Dims4 shape{1, 3, tensor_height, tensor_width};
    if (!context->setInputShape(slot.input_name.c_str(), shape)) {
      *error = "engine_input_shape_rejected";
      return false;
    }
    if (!context->setTensorAddress(slot.input_name.c_str(), workspace.device_input)) {
      *error = "engine_input_bind_failed";
      return false;
    }
    if (!context->setTensorAddress(slot.output_names[0].c_str(), device_rows)) {
      *error = "engine_output_bind_failed";
      return false;
    }
    if (slot.output_names.size() > 1 &&
        (device_extra == nullptr ||
         !context->setTensorAddress(slot.output_names[1].c_str(), device_extra))) {
      *error = "engine_prototype_bind_failed";
      return false;
    }
    if (!context->enqueueV3(workspace.stream)) {
      *error = "engine_enqueue_failed";
      return false;
    }
    if (copy_rows &&
        cudaMemcpyAsync(host_rows, device_rows, rows_capacity * sizeof(float),
                        cudaMemcpyDeviceToHost, workspace.stream) != cudaSuccess) {
      *error = "engine_output_copy_failed";
      return false;
    }
    if (slot.output_names.size() > 1 && host_extra != nullptr &&
        cudaMemcpyAsync(host_extra, device_extra, extra_capacity * sizeof(float),
                        cudaMemcpyDeviceToHost, workspace.stream) != cudaSuccess) {
      *error = "engine_prototype_copy_failed";
      return false;
    }
    return true;
  }
};

TrtPerception::TrtPerception() : impl_(std::make_unique<Impl>()) {}
TrtPerception::~TrtPerception() = default;

std::unique_ptr<TrtPerception> TrtPerception::load(const std::string& cache_dir,
                                                    std::string* error) {
  std::unique_ptr<TrtPerception> perception{new TrtPerception()};
  Impl& impl = *perception->impl_;
  impl.runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (impl.runtime == nullptr) {
    *error = "tensorrt_runtime_unavailable";
    return nullptr;
  }
  if (!impl.load_engine(cache_dir + "/pose.engine", &impl.pose, error) ||
      !impl.load_engine(cache_dir + "/person.engine", &impl.person, error) ||
      !impl.load_engine(cache_dir + "/bed.engine", &impl.bed, error)) {
    return nullptr;
  }
  if (!deepstream::CopyTelemetry::from_environment(&impl.copy_telemetry, error)) {
    return nullptr;
  }
  impl.workspaces.reserve(kInferenceWorkspaces);
  impl.pool.reserve(kInferenceWorkspaces);
  for (std::size_t index = 0; index < kInferenceWorkspaces; ++index) {
    auto workspace = std::make_unique<Workspace>();
    workspace->pose_context.reset(impl.pose.engine->createExecutionContext());
    workspace->person_context.reset(impl.person.engine->createExecutionContext());
    workspace->bed_context.reset(impl.bed.engine->createExecutionContext());
    if (workspace->pose_context == nullptr || workspace->person_context == nullptr ||
        workspace->bed_context == nullptr) {
      *error = "engine_context_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    if (cudaStreamCreate(&workspace->stream) != cudaSuccess) {
      *error = "cuda_stream_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    if (impl.copy_telemetry.enabled() &&
        (cudaEventCreate(&workspace->timing_start) != cudaSuccess ||
         cudaEventCreate(&workspace->timing_end) != cudaSuccess)) {
      *error = "copy telemetry: cuda event create failed";
      return nullptr;
    }
    const std::array<std::pair<float**, std::size_t>, 7> allocations{{
        {&workspace->device_input, kInputCapacity},
        {&workspace->device_pose, kPoseOutput},
        {&workspace->device_person, kPersonOutput},
        {&workspace->device_bed, kBedOutput},
        {&workspace->device_bed_prototypes, kBedPrototypes},
        {&workspace->device_pose_compact, kPoseOutput},
        {&workspace->device_person_compact, kPersonOutput},
    }};
    for (const auto& [pointer, capacity] : allocations) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer), capacity * sizeof(float)) != cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    if (cudaMalloc(reinterpret_cast<void**>(&workspace->device_bed_records),
                   kPostprocessTensorRows * sizeof(PackedBedRecord)) != cudaSuccess) {
      *error = "cuda_alloc_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    for (PostprocessChannelHeader** pointer :
         {&workspace->device_pose_header, &workspace->device_person_header,
          &workspace->device_bed_header}) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer), sizeof(PostprocessChannelHeader)) !=
          cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    for (std::uint64_t** pointer : {&workspace->bed_finalize.keys[0],
                                   &workspace->bed_finalize.keys[1]}) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer),
                     kBedFinalizeEntries * sizeof(std::uint64_t)) != cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    for (std::uint32_t** pointer : {&workspace->bed_finalize.points[0],
                                   &workspace->bed_finalize.points[1]}) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer),
                     kBedFinalizeEntries * sizeof(std::uint32_t)) != cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    const std::array<std::pair<void**, std::size_t>, 6> bed_finalize_allocations{{
        {reinterpret_cast<void**>(&workspace->bed_finalize.row_to_record),
         kBedFinalizeSegments * sizeof(std::int32_t)},
        {reinterpret_cast<void**>(&workspace->bed_finalize.active_count),
         kBedFinalizeSegments * sizeof(std::int32_t)},
        {reinterpret_cast<void**>(&workspace->bed_finalize.crop),
         kBedFinalizeSegments * sizeof(int4)},
        {reinterpret_cast<void**>(&workspace->bed_finalize.sum_y),
         kBedFinalizeSegments * sizeof(std::int64_t)},
        {reinterpret_cast<void**>(&workspace->bed_finalize.sum_x),
         kBedFinalizeSegments * sizeof(std::int64_t)},
        {reinterpret_cast<void**>(&workspace->bed_finalize.offsets),
         (kBedFinalizeSegments + 1) * sizeof(std::int32_t)},
    }};
    for (const auto& [pointer, bytes] : bed_finalize_allocations) {
      if (cudaMalloc(pointer, bytes) != cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    std::array<std::int32_t, kBedFinalizeSegments + 1> offsets{};
    for (int segment = 0; segment <= kBedFinalizeSegments; ++segment)
      offsets[segment] = segment * kBedFinalizePixels;
    if (cudaMemcpy(workspace->bed_finalize.offsets, offsets.data(), sizeof(offsets),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      *error = "cuda_alloc_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    if (!query_bed_finalize_workspace_temp_bytes(&workspace->bed_finalize, error)) return nullptr;
    if (cudaMalloc(&workspace->bed_finalize.cub_temp, workspace->bed_finalize.cub_temp_bytes) !=
        cudaSuccess) {
      *error = "cuda_alloc_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    workspace->host_pose.resize(kPoseOutput);
    workspace->host_person.resize(kPersonOutput);
    workspace->host_bed_records.resize(kPostprocessTensorRows);
    impl.pool.add(workspace.get());
    impl.workspaces.push_back(std::move(workspace));
  }
  return perception;
}

std::vector<std::string> TrtPerception::engine_names() const { return {"bed", "person", "pose"}; }

InferStatus TrtPerception::infer_host(const seeon::HostFrameView& frame, bool run_person_engine,
                                      PerceptionResult* result, std::string* error) {
  if (error == nullptr || result == nullptr || frame.rgba_host == nullptr || frame.width <= 0 ||
      frame.height <= 0 || frame.row_stride_bytes < static_cast<std::ptrdiff_t>(frame.width) * 4 ||
      frame.row_stride_bytes > std::numeric_limits<int>::max()) {
    if (error != nullptr) *error = "invalid_host_frame";
    return InferStatus::kFailed;
  }
  const auto affine = perception::letterbox_affine(frame.height, frame.width);
  const std::size_t tensor_size =
      3ULL * static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
  std::vector<float> staging_input(tensor_size);
  preprocess_rgba_to_bgr_tensor(frame.rgba_host, frame.width, frame.height,
                                static_cast<int>(frame.row_stride_bytes), affine,
                                staging_input.data());
  Workspace* workspace = impl_->pool.try_acquire();
  if (workspace == nullptr) return InferStatus::kDroppedBusy;
  const PoolLease<Workspace> lease{impl_->pool, *workspace};
  bool async_work_enqueued = false;
  bool complete = false;
  std::size_t pose_count = 0;
  std::size_t person_count = 0;
  std::size_t bed_count = 0;
  std::size_t transfer_bytes = 0;
  if (cudaMemcpyAsync(workspace->device_input, staging_input.data(), tensor_size * sizeof(float),
                      cudaMemcpyHostToDevice, workspace->stream) != cudaSuccess) {
    *error = "input_copy_failed";
    goto epilogue;
  }
  async_work_enqueued = true;
  if (!Impl::enqueue_engine(impl_->pose, workspace->pose_context.get(), *workspace,
                            affine.tensor_height, affine.tensor_width, workspace->device_pose,
                            kPoseOutput, nullptr, nullptr, 0, nullptr, false, error) ||
      !compact_pose_rows_device(workspace->device_pose, workspace->device_pose_compact,
                                workspace->device_pose_header, workspace->stream, error) ||
      (run_person_engine &&
       !Impl::enqueue_engine(impl_->person, workspace->person_context.get(), *workspace,
                             affine.tensor_height, affine.tensor_width, workspace->device_person,
                             kPersonOutput, nullptr, nullptr, 0, nullptr, false, error)) ||
      (run_person_engine &&
       !compact_person_rows_device(workspace->device_person, workspace->device_person_compact,
                                   workspace->device_person_header, workspace->stream, error)) ||
      !Impl::enqueue_engine(impl_->bed, workspace->bed_context.get(), *workspace,
                            affine.tensor_height, affine.tensor_width, workspace->device_bed,
                            kBedOutput, nullptr, workspace->device_bed_prototypes, kBedPrototypes,
                            nullptr, false, error) ||
      !finalize_bed_rows_device(
          workspace->device_bed, workspace->device_bed_prototypes, workspace->device_bed_records,
          workspace->device_bed_header, &workspace->bed_finalize,
          BedFinalizeGeometry{frame.height, frame.width, affine.tensor_height, affine.tensor_width,
                              affine.gain, affine.box_pad_x, affine.box_pad_y,
                              affine.keypoint_pad_x, affine.keypoint_pad_y},
          workspace->stream, error)) {
    goto epilogue;
  }
  if (!run_person_engine &&
      cudaMemsetAsync(workspace->device_person_header, 0, sizeof(PostprocessChannelHeader),
                      workspace->stream) != cudaSuccess) {
    *error = "postprocess_person_skipped_header_failed";
    goto epilogue;
  }
  if (cudaMemcpyAsync(&workspace->host_pose_header, workspace->device_pose_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      workspace->stream) != cudaSuccess ||
      cudaMemcpyAsync(&workspace->host_person_header, workspace->device_person_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      workspace->stream) != cudaSuccess ||
      cudaMemcpyAsync(&workspace->host_bed_header, workspace->device_bed_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      workspace->stream) != cudaSuccess) {
    *error = "postprocess_header_copy_failed";
    goto epilogue;
  }
  if (cudaStreamSynchronize(workspace->stream) != cudaSuccess) {
    *error = "postprocess_header_sync_failed";
    goto epilogue;
  }
  if (!validate_postprocess_channel_headers(
          workspace->host_pose_header, workspace->host_person_header, workspace->host_bed_header,
          run_person_engine, error))
    goto epilogue;
  pose_count = static_cast<std::size_t>(workspace->host_pose_header.count);
  person_count = run_person_engine
                     ? static_cast<std::size_t>(workspace->host_person_header.count)
                     : 0;
  bed_count = static_cast<std::size_t>(workspace->host_bed_header.count);
  if (!postprocess_transfer_bytes(pose_count, person_count, bed_count, &transfer_bytes)) {
    *error = "postprocess_header_invalid";
    goto epilogue;
  }
  if (cudaMemcpyAsync(workspace->host_pose.data(), workspace->device_pose_compact,
                      pose_count * kPostprocessPoseRowStride * sizeof(float),
                      cudaMemcpyDeviceToHost, workspace->stream) != cudaSuccess ||
      (run_person_engine &&
       cudaMemcpyAsync(workspace->host_person.data(), workspace->device_person_compact,
                       person_count * kPostprocessPersonRowStride * sizeof(float),
                       cudaMemcpyDeviceToHost, workspace->stream) != cudaSuccess) ||
      cudaMemcpyAsync(workspace->host_bed_records.data(), workspace->device_bed_records,
                      bed_count * sizeof(PackedBedRecord), cudaMemcpyDeviceToHost,
                      workspace->stream) != cudaSuccess) {
    *error = "postprocess_rows_copy_failed";
    goto epilogue;
  }
  complete = true;

epilogue:
  if (async_work_enqueued && cudaStreamSynchronize(workspace->stream) != cudaSuccess) {
    *error = "engine_stream_sync_failed";
    complete = false;
  }
  if (!complete) return InferStatus::kFailed;
  result->pose = perception::parse_pose_rows(
      std::span<const float>{workspace->host_pose.data(),
                             pose_count * kPostprocessPoseRowStride},
      affine);
  if (run_person_engine) {
    result->person = perception::parse_person_rows(
        std::span<const float>{workspace->host_person.data(),
                               person_count * kPostprocessPersonRowStride},
        affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed.clear();
  result->bed.reserve(bed_count);
  for (std::size_t index = 0; index < bed_count; ++index) {
    const PackedBedRecord& packed = workspace->host_bed_records[index];
    if (packed.point_count < 0 || packed.point_count > kPostprocessBedMaxPoints) {
      *error = "bed_postprocess_record_invalid";
      return InferStatus::kFailed;
    }
    perception::ParsedBedRegion region{
        perception::ParsedBox{packed.box[0], packed.box[1], packed.box[2], packed.box[3],
                               packed.confidence},
        {}};
    region.polygon.reserve(static_cast<std::size_t>(packed.point_count));
    for (int point = 0; point < packed.point_count; ++point)
      region.polygon.emplace_back(packed.points[point][0], packed.points[point][1]);
    result->bed.push_back(std::move(region));
  }
  result->source_width = frame.width;
  result->source_height = frame.height;
  return InferStatus::kCompleted;
}

InferStatus TrtPerception::infer_device(std::string_view camera_id,
                                        const seeon::DeviceFrameView& frame,
                                        bool run_person_engine, PerceptionResult* result,
                                        std::string* error) {
  if (error == nullptr || result == nullptr || frame.rgba_device == nullptr || frame.width <= 0 ||
      frame.height <= 0 || frame.pitch_bytes < static_cast<std::size_t>(frame.width) * 4 ||
      frame.device_ordinal < 0) {
    if (error != nullptr) *error = "invalid_device_frame";
    return InferStatus::kFailed;
  }
  const bool telemetry_enabled = impl_->copy_telemetry.enabled();
  const auto pool_wait_started =
      telemetry_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  Workspace* available = impl_->pool.try_acquire();
  double pool_wait_us = 0.0;
  if (telemetry_enabled) {
    pool_wait_us = std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - pool_wait_started)
                       .count();
  }
  const auto box_source = run_person_engine ? deepstream::CopyTelemetry::BoxSource::kPerson
                                            : deepstream::CopyTelemetry::BoxSource::kPose;
  if (available == nullptr) {
    if (telemetry_enabled &&
        !impl_->copy_telemetry.record_busy_surface_drop(camera_id, box_source, error)) {
      return InferStatus::kFailed;
    }
    return InferStatus::kDroppedBusy;
  }
  const PoolLease<Workspace> lease{impl_->pool, *available};
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != frame.device_ordinal) {
    *error = "device_ordinal_invalid";
    return InferStatus::kFailed;
  }
  const auto affine = perception::letterbox_affine(frame.height, frame.width);
  bool async_work_enqueued = false;
  bool complete = false;
  std::size_t pose_count = 0;
  std::size_t person_count = 0;
  std::size_t bed_count = 0;
  std::size_t transfer_bytes = 0;
  float gpu_ms = 0.0F;
  // The preprocess helper may have submitted work before reporting a launch
  // failure. Its contract requires the borrowed frame and workspace to remain
  // live until this stream is synchronized.
  async_work_enqueued = true;
  if (telemetry_enabled &&
      cudaEventRecord(available->timing_start, available->stream) != cudaSuccess) {
    *error = "copy telemetry: cuda event record failed";
    goto epilogue;
  }
  if (!preprocess_rgba_device_to_bgr_tensor(frame.rgba_device, frame.width, frame.height,
                                            frame.pitch_bytes, affine, available->device_input,
                                            available->stream, error)) {
    goto epilogue;
  }
  if (!Impl::enqueue_engine(impl_->pose, available->pose_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_pose,
                            kPoseOutput, available->host_pose.data(), nullptr, 0, nullptr, false,
                            error) ||
      !compact_pose_rows_device(available->device_pose, available->device_pose_compact,
                                available->device_pose_header, available->stream, error)) {
    goto epilogue;
  }
  if (run_person_engine &&
      !Impl::enqueue_engine(impl_->person, available->person_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_person,
                            kPersonOutput, available->host_person.data(), nullptr, 0, nullptr,
                            false, error)) {
    goto epilogue;
  }
  if (run_person_engine &&
      !compact_person_rows_device(available->device_person, available->device_person_compact,
                                  available->device_person_header, available->stream, error)) {
    goto epilogue;
  }
  if (!Impl::enqueue_engine(impl_->bed, available->bed_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_bed,
                            kBedOutput, nullptr, available->device_bed_prototypes,
                            kBedPrototypes, nullptr, false, error) ||
      !finalize_bed_rows_device(
          available->device_bed, available->device_bed_prototypes, available->device_bed_records,
          available->device_bed_header, &available->bed_finalize,
          BedFinalizeGeometry{frame.height, frame.width, affine.tensor_height, affine.tensor_width,
                              affine.gain, affine.box_pad_x, affine.box_pad_y,
                              affine.keypoint_pad_x, affine.keypoint_pad_y},
          available->stream, error)) {
    goto epilogue;
  }
  if (!run_person_engine &&
      cudaMemsetAsync(available->device_person_header, 0, sizeof(PostprocessChannelHeader),
                      available->stream) != cudaSuccess) {
    *error = "postprocess_person_skipped_header_failed";
    goto epilogue;
  }
  if (telemetry_enabled &&
      cudaEventRecord(available->timing_end, available->stream) != cudaSuccess) {
    *error = "copy telemetry: cuda event record failed";
    goto epilogue;
  }
  if (cudaMemcpyAsync(&available->host_pose_header, available->device_pose_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      available->stream) != cudaSuccess ||
      cudaMemcpyAsync(&available->host_person_header, available->device_person_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      available->stream) != cudaSuccess ||
      cudaMemcpyAsync(&available->host_bed_header, available->device_bed_header,
                      sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost,
                      available->stream) != cudaSuccess) {
    *error = "postprocess_header_copy_failed";
    goto epilogue;
  }
  if (cudaStreamSynchronize(available->stream) != cudaSuccess) {
    *error = "postprocess_header_sync_failed";
    goto epilogue;
  }
  if (telemetry_enabled &&
      cudaEventElapsedTime(&gpu_ms, available->timing_start, available->timing_end) !=
          cudaSuccess) {
    *error = "copy telemetry: cuda event elapsed failed";
    goto epilogue;
  }
  if (!validate_postprocess_channel_headers(
          available->host_pose_header, available->host_person_header, available->host_bed_header,
          run_person_engine, error))
    goto epilogue;
  pose_count = static_cast<std::size_t>(available->host_pose_header.count);
  person_count = run_person_engine
                     ? static_cast<std::size_t>(available->host_person_header.count)
                     : 0;
  bed_count = static_cast<std::size_t>(available->host_bed_header.count);
  if (!postprocess_transfer_bytes(pose_count, person_count, bed_count, &transfer_bytes)) {
    *error = "postprocess_header_invalid";
    goto epilogue;
  }
  if (cudaMemcpyAsync(available->host_pose.data(), available->device_pose_compact,
                      pose_count * kPostprocessPoseRowStride * sizeof(float),
                      cudaMemcpyDeviceToHost, available->stream) != cudaSuccess ||
      (run_person_engine &&
       cudaMemcpyAsync(available->host_person.data(), available->device_person_compact,
                       person_count * kPostprocessPersonRowStride * sizeof(float),
                       cudaMemcpyDeviceToHost, available->stream) != cudaSuccess) ||
      cudaMemcpyAsync(available->host_bed_records.data(), available->device_bed_records,
                      bed_count * sizeof(PackedBedRecord), cudaMemcpyDeviceToHost,
                      available->stream) != cudaSuccess) {
    *error = "postprocess_rows_copy_failed";
    goto epilogue;
  }
  complete = true;

epilogue:
  if (async_work_enqueued && cudaStreamSynchronize(available->stream) != cudaSuccess) {
    *error = "engine_stream_sync_failed";
    complete = false;
  }
  if (!complete) return InferStatus::kFailed;
  result->pose = perception::parse_pose_rows(
      std::span<const float>{available->host_pose.data(),
                             pose_count * kPostprocessPoseRowStride},
      affine);
  if (run_person_engine) {
    result->person = perception::parse_person_rows(
        std::span<const float>{available->host_person.data(),
                               person_count * kPostprocessPersonRowStride},
        affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed.clear();
  result->bed.reserve(bed_count);
  for (std::size_t index = 0; index < bed_count; ++index) {
    const PackedBedRecord& packed = available->host_bed_records[index];
    if (packed.point_count < 0 || packed.point_count > kPostprocessBedMaxPoints) {
      *error = "bed_postprocess_record_invalid";
      return InferStatus::kFailed;
    }
    perception::ParsedBedRegion region{
        perception::ParsedBox{packed.box[0], packed.box[1], packed.box[2], packed.box[3],
                               packed.confidence},
        {}};
    region.polygon.reserve(static_cast<std::size_t>(packed.point_count));
    for (int point = 0; point < packed.point_count; ++point)
      region.polygon.emplace_back(packed.points[point][0], packed.points[point][1]);
    result->bed.push_back(std::move(region));
  }
  result->source_width = frame.width;
  result->source_height = frame.height;
  if (telemetry_enabled &&
      !impl_->copy_telemetry.record_completed_frame(
          deepstream::CopyTelemetry::CompletedFrame{
              camera_id, 0, static_cast<std::uint64_t>(transfer_bytes), box_source, pool_wait_us,
              static_cast<double>(gpu_ms) * 1000.0},
          error)) {
    return InferStatus::kFailed;
  }
  return InferStatus::kCompleted;
}

}  // namespace seeon::trt
