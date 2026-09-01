#include "trt_perception.hpp"

#include "preprocess_gpu.hpp"
#include "source_runtime.hpp"
#include "workspace_pool.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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
  float* device_input = nullptr;
  float* device_pose = nullptr;
  float* device_person = nullptr;
  float* device_bed = nullptr;
  float* device_bed_prototypes = nullptr;
  std::vector<float> host_pose;
  std::vector<float> host_person;
  std::vector<float> host_bed;
  std::vector<float> host_bed_prototypes;

  ~Workspace() {
    for (float* pointer : {device_input, device_pose, device_person, device_bed,
                           device_bed_prototypes}) {
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
                             float* host_extra, std::string* error) {
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
    if (cudaMemcpyAsync(host_rows, device_rows, rows_capacity * sizeof(float),
                        cudaMemcpyDeviceToHost, workspace.stream) != cudaSuccess) {
      *error = "engine_output_copy_failed";
      return false;
    }
    if (slot.output_names.size() > 1 &&
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
    const std::array<std::pair<float**, std::size_t>, 5> allocations{{
        {&workspace->device_input, kInputCapacity},
        {&workspace->device_pose, kPoseOutput},
        {&workspace->device_person, kPersonOutput},
        {&workspace->device_bed, kBedOutput},
        {&workspace->device_bed_prototypes, kBedPrototypes},
    }};
    for (const auto& [pointer, capacity] : allocations) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer), capacity * sizeof(float)) != cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    workspace->host_pose.resize(kPoseOutput);
    workspace->host_person.resize(kPersonOutput);
    workspace->host_bed.resize(kBedOutput);
    workspace->host_bed_prototypes.resize(kBedPrototypes);
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
  if (cudaMemcpyAsync(workspace->device_input, staging_input.data(), tensor_size * sizeof(float),
                      cudaMemcpyHostToDevice, workspace->stream) != cudaSuccess) {
    *error = "input_copy_failed";
    goto epilogue;
  }
  async_work_enqueued = true;
  if (!Impl::enqueue_engine(impl_->pose, workspace->pose_context.get(), *workspace,
                            affine.tensor_height, affine.tensor_width, workspace->device_pose,
                            kPoseOutput, workspace->host_pose.data(), nullptr, 0, nullptr, error) ||
      (run_person_engine &&
       !Impl::enqueue_engine(impl_->person, workspace->person_context.get(), *workspace,
                             affine.tensor_height, affine.tensor_width, workspace->device_person,
                             kPersonOutput, workspace->host_person.data(), nullptr, 0, nullptr,
                             error)) ||
      !Impl::enqueue_engine(impl_->bed, workspace->bed_context.get(), *workspace,
                            affine.tensor_height, affine.tensor_width, workspace->device_bed,
                            kBedOutput, workspace->host_bed.data(),
                            workspace->device_bed_prototypes, kBedPrototypes,
                            workspace->host_bed_prototypes.data(), error)) {
    goto epilogue;
  }
  complete = true;

epilogue:
  if (async_work_enqueued && cudaStreamSynchronize(workspace->stream) != cudaSuccess) {
    *error = "engine_stream_sync_failed";
    complete = false;
  }
  if (!complete) return InferStatus::kFailed;
  const std::vector<double> pose_rows{workspace->host_pose.begin(), workspace->host_pose.end()};
  const std::vector<double> bed_rows{workspace->host_bed.begin(), workspace->host_bed.end()};
  const std::vector<double> prototypes{workspace->host_bed_prototypes.begin(),
                                       workspace->host_bed_prototypes.end()};
  result->pose = perception::parse_pose_rows(pose_rows, affine);
  if (run_person_engine) {
    const std::vector<double> person_rows{workspace->host_person.begin(),
                                           workspace->host_person.end()};
    result->person = perception::parse_person_rows(person_rows, affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed = perception::parse_bed_rows(bed_rows, prototypes, affine, kBedConfidence);
  result->source_width = frame.width;
  result->source_height = frame.height;
  return InferStatus::kCompleted;
}

InferStatus TrtPerception::infer_device(const seeon::DeviceFrameView& frame,
                                        bool run_person_engine, PerceptionResult* result,
                                        std::string* error) {
  if (error == nullptr || result == nullptr || frame.rgba_device == nullptr || frame.width <= 0 ||
      frame.height <= 0 || frame.pitch_bytes < static_cast<std::size_t>(frame.width) * 4 ||
      frame.device_ordinal < 0) {
    if (error != nullptr) *error = "invalid_device_frame";
    return InferStatus::kFailed;
  }
  Workspace* available = impl_->pool.try_acquire();
  if (available == nullptr) return InferStatus::kDroppedBusy;
  const PoolLease<Workspace> lease{impl_->pool, *available};
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != frame.device_ordinal) {
    *error = "device_ordinal_invalid";
    return InferStatus::kFailed;
  }
  const auto affine = perception::letterbox_affine(frame.height, frame.width);
  bool async_work_enqueued = false;
  bool complete = false;
  // The preprocess helper may have submitted work before reporting a launch
  // failure. Its contract requires the borrowed frame and workspace to remain
  // live until this stream is synchronized.
  async_work_enqueued = true;
  if (!preprocess_rgba_device_to_bgr_tensor(frame.rgba_device, frame.width, frame.height,
                                            frame.pitch_bytes, affine, available->device_input,
                                            available->stream, error)) {
    goto epilogue;
  }
  if (!Impl::enqueue_engine(impl_->pose, available->pose_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_pose,
                            kPoseOutput, available->host_pose.data(), nullptr, 0, nullptr, error)) {
    goto epilogue;
  }
  if (run_person_engine &&
      !Impl::enqueue_engine(impl_->person, available->person_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_person,
                            kPersonOutput, available->host_person.data(), nullptr, 0, nullptr, error)) {
    goto epilogue;
  }
  if (!Impl::enqueue_engine(impl_->bed, available->bed_context.get(), *available,
                            affine.tensor_height, affine.tensor_width, available->device_bed,
                            kBedOutput, available->host_bed.data(), available->device_bed_prototypes,
                            kBedPrototypes, available->host_bed_prototypes.data(), error)) {
    goto epilogue;
  }
  complete = true;

epilogue:
  if (async_work_enqueued && cudaStreamSynchronize(available->stream) != cudaSuccess) {
    *error = "engine_stream_sync_failed";
    complete = false;
  }
  if (!complete) return InferStatus::kFailed;
  const std::vector<double> pose_rows{available->host_pose.begin(), available->host_pose.end()};
  const std::vector<double> bed_rows{available->host_bed.begin(), available->host_bed.end()};
  const std::vector<double> prototypes{available->host_bed_prototypes.begin(),
                                       available->host_bed_prototypes.end()};
  result->pose = perception::parse_pose_rows(pose_rows, affine);
  if (run_person_engine) {
    const std::vector<double> person_rows{available->host_person.begin(),
                                           available->host_person.end()};
    result->person = perception::parse_person_rows(person_rows, affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed = perception::parse_bed_rows(bed_rows, prototypes, affine, kBedConfidence);
  result->source_width = frame.width;
  result->source_height = frame.height;
  return InferStatus::kCompleted;
}

}  // namespace seeon::trt
