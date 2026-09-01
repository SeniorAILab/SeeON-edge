// TensorRT-free stub used ONLY by host/CTest targets that exercise command and
// lifecycle logic without a GPU. The production child and the preflight binary
// always link the real trt_perception.cpp.

#include "trt_perception.hpp"

namespace seeon::trt {

class TrtPerception::Impl {};

TrtPerception::TrtPerception() : impl_(nullptr) {}
TrtPerception::~TrtPerception() = default;

std::unique_ptr<TrtPerception> TrtPerception::load(const std::string& cache_dir,
                                                   std::string* error) {
  static_cast<void>(cache_dir);
  *error = "tensorrt_unavailable_in_stub_build";
  return nullptr;
}

InferStatus TrtPerception::infer_host(const seeon::HostFrameView& frame,
                                      bool run_person_engine, PerceptionResult* result,
                                      std::string* error) {
  static_cast<void>(frame);
  static_cast<void>(run_person_engine);
  static_cast<void>(result);
  *error = "tensorrt_unavailable_in_stub_build";
  return InferStatus::kFailed;
}

InferStatus TrtPerception::infer_device(const seeon::DeviceFrameView& frame,
                                        bool run_person_engine, PerceptionResult* result,
                                        std::string* error) {
  static_cast<void>(frame);
  static_cast<void>(run_person_engine);
  static_cast<void>(result);
  *error = "tensorrt_unavailable_in_stub_build";
  return InferStatus::kFailed;
}

std::vector<std::string> TrtPerception::engine_names() const { return {}; }

}  // namespace seeon::trt
