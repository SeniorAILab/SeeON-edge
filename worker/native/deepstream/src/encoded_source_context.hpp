#pragma once

#include "source_runtime.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace seeon {
struct EncodedSourceContext {
  FrameCallback frames;
  FailureCallback failures;
  AccessUnitCallback access_units;
  std::string camera;
  GstElement* pipeline;
  std::uint64_t order = 0;
  bool linked = false;
  std::vector<std::uint8_t> codec_data;
  GstElement* preview_valve = nullptr;
  GstElement* decode_queue = nullptr;
  std::atomic<std::uint64_t> preview_encoded{0};
  std::atomic<std::uint64_t> preview_viewers{0};
  std::optional<ParsedAccessUnit> pending_duration;
  std::mutex preview_mutex;
  std::condition_variable preview_ready;
};

GstFlowReturn on_encoded_sample(GstAppSink* sink, gpointer raw);
}  // namespace seeon
#endif
