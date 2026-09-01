#pragma once

// TensorRT-backed perception stage for the DeepStream child.
//
// Owns the three pinned engines (pose/person/bed) from the content-addressed
// engine cache, the C3-parity preprocessing (letterbox 640-side, stride 32,
// pad 114, BGR planes, FP32/255 -- see parity/preprocess.py), and the parse +
// association stage (native_perception.hpp). One instance per child process.

#include "native_perception.hpp"
#include "preprocess_cpu.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace seeon {
struct HostFrameView;
struct DeviceFrameView;
}  // namespace seeon

namespace seeon::trt {

struct PerceptionResult {
  perception::ParsedPose pose;
  std::vector<perception::ParsedBox> person;
  std::vector<perception::ParsedBedRegion> bed;
  int source_width = 0;
  int source_height = 0;
};

struct EngineIdentity {
  std::string name;
  std::string path;
};

// Confidence thresholds mirror the shipped Python adapters: pose keeps its own
// strict >0.05 rule inside the parser; person/bed use the shipped 0.25 default
// (worker/adapters/model/yolo_pose.py conf handling is proven in C3 evidence).
inline constexpr double kPersonConfidence = 0.25;
inline constexpr double kBedConfidence = 0.25;

enum class InferStatus : std::uint8_t {
  kCompleted,
  kDroppedBusy,
  kFailed,
};

class TrtPerception {
 public:
  ~TrtPerception();
  TrtPerception(const TrtPerception&) = delete;
  TrtPerception& operator=(const TrtPerception&) = delete;

  // Loads pose.engine / person.engine / bed.engine from cache_dir. Returns
  // nullptr and fills *error on any failure (missing file, deserialize
  // failure, tensor-name mismatch).
  static std::unique_ptr<TrtPerception> load(const std::string& cache_dir,
                                             std::string* error);

  // Runs synchronously: completion or failure is known before this returns.
  // Host input is only for deterministic warmup and explicitly selected
  // non-RTSP paths. Device inference never falls back to host input.
  [[nodiscard]] InferStatus infer_host(const seeon::HostFrameView& frame,
                                       bool run_person_engine,
                                       PerceptionResult* result,
                                       std::string* error);
  [[nodiscard]] InferStatus infer_device(const seeon::DeviceFrameView& frame,
                                         bool run_person_engine,
                                         PerceptionResult* result,
                                         std::string* error);

  [[nodiscard]] std::vector<std::string> engine_names() const;

 private:
  TrtPerception();
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace seeon::trt
