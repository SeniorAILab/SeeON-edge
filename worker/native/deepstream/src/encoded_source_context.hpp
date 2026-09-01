#pragma once

#include "source_runtime.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#ifdef SEEON_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <optional>
#include <string>
#include <vector>
#endif

namespace seeon {
struct InFlightGate {
  [[nodiscard]] bool admit() {
    std::lock_guard lock{mutex};
    if (!accepting) return false;
    ++in_flight;
    return true;
  }

  void release() {
    std::lock_guard lock{mutex};
    --in_flight;
    if (in_flight == 0) drained.notify_all();
  }

  void stop() {
    std::lock_guard lock{mutex};
    accepting = false;
  }

  [[nodiscard]] bool wait_drained(const std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock{mutex};
    return drained.wait_until(lock, deadline, [this] { return in_flight == 0; });
  }

 private:
  std::mutex mutex;
  std::condition_variable drained;
  bool accepting = true;
  std::uint64_t in_flight = 0;
};

class InFlightLease {
 public:
  explicit InFlightLease(const std::shared_ptr<InFlightGate>& gate) : gate_(gate) {
    admitted_ = gate_ != nullptr && gate_->admit();
  }
  ~InFlightLease() {
    if (admitted_) gate_->release();
  }
  InFlightLease(const InFlightLease&) = delete;
  InFlightLease& operator=(const InFlightLease&) = delete;
  [[nodiscard]] explicit operator bool() const { return admitted_; }

 private:
  std::shared_ptr<InFlightGate> gate_;
  bool admitted_ = false;
};

#ifdef SEEON_HAS_GSTREAMER
struct EncodedSourceContext {
  DeviceFrameCallback frames;
  FailureCallback failures;
  AccessUnitCallback access_units;
  PreviewCallback previews;
  PipelineBindingPtr binding;
  std::shared_ptr<InFlightGate> gate;
  std::string camera;
  GstElement* pipeline;
  std::uint64_t order = 0;
  bool linked = false;
  std::vector<std::uint8_t> codec_data;
  GstElement* preview_valve = nullptr;
  GstElement* decode_queue = nullptr;
  std::atomic<std::uint64_t> preview_encoded{0};
  std::atomic<std::uint64_t> preview_viewers{0};
  std::atomic<std::uint64_t> au_forwarded{0};
  std::vector<std::uint8_t> last_preview_jpeg;  // guarded by preview_mutex
  std::optional<ParsedAccessUnit> pending_duration;
  std::int64_t last_duration = 0;
  std::atomic<bool> parser_failure_latched{false};
  std::atomic<bool> inference_failure_latched{false};
  std::mutex sample_mutex;
  std::mutex preview_mutex;
  std::condition_variable preview_ready;
};

GstFlowReturn on_encoded_sample(GstAppSink* sink, gpointer raw);
void flush_pending_access_unit(EncodedSourceContext* context);
#endif
}  // namespace seeon
