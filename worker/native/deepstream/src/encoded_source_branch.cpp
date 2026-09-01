#include "encoded_source_branch.hpp"

#include "encoded_source_bus.hpp"
#include "encoded_source_context.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <glib.h>
#include <cuda_runtime_api.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <ranges>
#include <utility>

namespace seeon {
namespace {
constexpr int kExpectedGpuId = 0;

class SampleLease {
 public:
  explicit SampleLease(GstSample* sample) : sample_(sample) {}
  ~SampleLease() {
    if (sample_ != nullptr) gst_sample_unref(sample_);
  }
  SampleLease(const SampleLease&) = delete;
  SampleLease& operator=(const SampleLease&) = delete;
  [[nodiscard]] GstSample* get() const { return sample_; }

 private:
  GstSample* sample_;
};

class BufferMap {
 public:
  explicit BufferMap(GstBuffer* buffer) : buffer_(buffer), mapped_(gst_buffer_map(buffer, &info_, GST_MAP_READ)) {}
  ~BufferMap() {
    if (mapped_) gst_buffer_unmap(buffer_, &info_);
  }
  BufferMap(const BufferMap&) = delete;
  BufferMap& operator=(const BufferMap&) = delete;
  [[nodiscard]] bool mapped() const { return mapped_; }
  [[nodiscard]] const GstMapInfo& info() const { return info_; }

 private:
  GstBuffer* buffer_;
  GstMapInfo info_{};
  bool mapped_;
};

bool is_nvmm_rgba_caps(const GstCaps* caps) {
  if (caps == nullptr || gst_caps_get_size(caps) != 1) return false;
  const GstCapsFeatures* features = gst_caps_get_features(caps, 0);
  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  const char* format = structure == nullptr ? nullptr : gst_structure_get_string(structure, "format");
  return features != nullptr && gst_caps_features_contains(features, "memory:NVMM") &&
         format != nullptr && g_str_equal(format, "RGBA");
}

void inference_contract_failure(EncodedSourceContext* context) {
  if (!context->inference_failure_latched.exchange(true)) {
    context->failures({context->camera, "nvmm_surface_contract", FailureScope::kFatal});
  }
}

GstPadProbeReturn on_preview_encoded(GstPad*, GstPadProbeInfo* info, gpointer raw) {
  auto* context = static_cast<EncodedSourceContext*>(raw);
  GstBuffer* buffer = gst_pad_probe_info_get_buffer(info);
  GstMapInfo mapped{};
  if (!context->previews || !gst_buffer_map(buffer, &mapped, GST_MAP_READ)) {
    return GST_PAD_PROBE_OK;
  }
  context->previews(
      context->camera, GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0,
      {mapped.data, mapped.data + mapped.size});
  {
    std::lock_guard lock{context->preview_mutex};
    context->last_preview_jpeg.assign(mapped.data, mapped.data + mapped.size);
    ++context->preview_encoded;
  }
  gst_buffer_unmap(buffer, &mapped);
  context->preview_ready.notify_all();
  return GST_PAD_PROBE_OK;
}

GstFlowReturn on_inference_sample(GstAppSink* sink, gpointer raw) {
  auto* context = static_cast<EncodedSourceContext*>(raw);
  SampleLease sample{gst_app_sink_pull_sample(sink)};
  if (sample.get() == nullptr) return GST_FLOW_ERROR;
  InFlightLease callback{context->gate};
  if (!callback) return GST_FLOW_FLUSHING;
  GstCaps* caps = gst_sample_get_caps(sample.get());
  GstBuffer* buffer = gst_sample_get_buffer(sample.get());
  if (!is_nvmm_rgba_caps(caps) || buffer == nullptr) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  const GstStructure* caps_structure = gst_caps_get_structure(caps, 0);
  gint caps_width = 0;
  gint caps_height = 0;
  if (caps_structure == nullptr ||
      !gst_structure_get_int(caps_structure, "width", &caps_width) ||
      !gst_structure_get_int(caps_structure, "height", &caps_height) || caps_width <= 0 ||
      caps_height <= 0) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  BufferMap descriptor{buffer};
  if (!descriptor.mapped() || descriptor.info().size < sizeof(NvBufSurface)) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  const auto* surface = reinterpret_cast<const NvBufSurface*>(descriptor.info().data);
  if (surface == nullptr || surface->surfaceList == nullptr || surface->batchSize != 1 ||
      surface->numFilled != 1 ||
      surface->memType != NVBUF_MEM_CUDA_DEVICE || surface->gpuId != kExpectedGpuId) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  const NvBufSurfaceParams& params = surface->surfaceList[0];
  if (params.dataPtr == nullptr || params.width != static_cast<guint>(caps_width) ||
      params.height != static_cast<guint>(caps_height) ||
      params.pitch < params.width * 4U || params.colorFormat != NVBUF_COLOR_FORMAT_RGBA ||
      params.layout != NVBUF_LAYOUT_PITCH) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, params.dataPtr) != cudaSuccess) {
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
#if CUDART_VERSION >= 10000
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != kExpectedGpuId) {
#else
  if (attributes.memoryType != cudaMemoryTypeDevice || attributes.device != kExpectedGpuId) {
#endif
    inference_contract_failure(context);
    return GST_FLOW_ERROR;
  }
  const DeviceFrameView view{
      {GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0,
       static_cast<std::uint64_t>(g_get_real_time()) * 1000ULL},
      static_cast<int>(params.width),
      static_cast<int>(params.height),
      params.pitch,
      params.dataPtr,
      static_cast<int>(surface->gpuId),
  };
  context->frames(context->camera, context->binding, view);
  return GST_FLOW_OK;
}

bool verify_int_property(GstElement* item, const char* name, int expected) {
  GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(item), name);
  if (spec == nullptr) return false;
  GValue value = G_VALUE_INIT;
  g_value_init(&value, G_PARAM_SPEC_VALUE_TYPE(spec));
  g_object_get_property(G_OBJECT(item), name, &value);
  const bool matches = G_VALUE_HOLDS_ENUM(&value) ? g_value_get_enum(&value) == expected
                       : G_VALUE_HOLDS_INT(&value) ? g_value_get_int(&value) == expected
                       : G_VALUE_HOLDS_UINT(&value) ? g_value_get_uint(&value) ==
                                                         static_cast<unsigned int>(expected)
                                                    : false;
  g_value_unset(&value);
  return matches;
}

GstElement* element(const char* factory) { return gst_element_factory_make(factory, nullptr); }

void on_rtp_pad(GstElement*, GstPad* output, gpointer raw) {
  auto* context = static_cast<EncodedSourceContext*>(raw);
  if (context->linked) return;
  GstCaps* input_caps = gst_pad_get_current_caps(output);
  const GstStructure* structure = input_caps == nullptr ? nullptr : gst_caps_get_structure(input_caps, 0);
  const char* encoding = structure == nullptr ? nullptr : gst_structure_get_string(structure, "encoding-name");
  const bool h264 = encoding != nullptr && g_ascii_strcasecmp(encoding, "H264") == 0;
  const bool h265 = encoding != nullptr && g_ascii_strcasecmp(encoding, "H265") == 0;
  if (!h264 && !h265) {
    if (input_caps != nullptr) gst_caps_unref(input_caps);
    return;
  }
  GstElement* depay = element(h264 ? "rtph264depay" : "rtph265depay");
  GstElement* parser = element(h264 ? "h264parse" : "h265parse");
  GstElement* parser_caps = element("capsfilter");
  GstElement* tee = element("tee");
  GstElement* record_queue = element("queue");
  GstElement* record_sink = element("appsink");
  GstElement* decode_queue = element("queue");
  GstElement* decoder = element("nvv4l2decoder");
  GstElement* convert = element("nvvideoconvert");
  GstElement* decoded_caps = element("capsfilter");
  GstElement* transform = element("seeonperceptiontransform");
  GstElement* decoded_tee = element("tee");
  GstElement* inference_queue = element("queue");
  GstElement* inference_convert = element("nvvideoconvert");
  GstElement* inference_caps = element("capsfilter");
  GstElement* sink = element("appsink");
  GstElement* preview_valve = element("valve");
  GstElement* preview_queue = element("queue");
  GstElement* preview_convert = element("nvvideoconvert");
  GstElement* preview_caps = element("capsfilter");
  GstElement* osd = element("nvdsosd");
  GstElement* jpeg_convert = element("nvvideoconvert");
  GstElement* jpeg_caps = element("capsfilter");
  GstElement* jpeg = element("nvjpegenc");
  GstElement* preview_sink = element("fakesink");
  const std::array<GstElement*, 24> elements{
      depay, parser, parser_caps, tee, record_queue, record_sink, decode_queue, decoder, convert,
      decoded_caps, transform, decoded_tee, inference_queue, inference_convert, inference_caps, sink,
      preview_valve, preview_queue, preview_convert, preview_caps, osd, jpeg_convert,
      jpeg_caps, jpeg};
  if (std::ranges::any_of(elements, [](GstElement* item) { return item == nullptr; }) ||
      preview_sink == nullptr) {
    context->failures({context->camera, "element_unavailable", FailureScope::kFatal});
    for (GstElement* item : elements) {
      if (item != nullptr) {
        static_cast<void>(gst_object_ref_sink(item));
        gst_object_unref(item);
      }
    }
    if (preview_sink != nullptr) {
      static_cast<void>(gst_object_ref_sink(preview_sink));
      gst_object_unref(preview_sink);
    }
    if (input_caps != nullptr) gst_caps_unref(input_caps);
    return;
  }
  g_object_set(parser, "config-interval", -1, nullptr);
  GstCaps* aligned_caps = gst_caps_from_string(
      h264 ? "video/x-h264,alignment=au,stream-format=byte-stream"
           : "video/x-h265,alignment=au,stream-format=byte-stream");
  g_object_set(parser_caps, "caps", aligned_caps, nullptr);
  gst_caps_unref(aligned_caps);
  g_object_set(record_queue, "max-size-buffers", 128U, "max-size-bytes", kMaxAuFrameBytes,
               nullptr);
  g_object_set(record_sink, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 128U,
               "drop", FALSE, nullptr);
  g_object_set(decode_queue, "max-size-buffers", 64U, "max-size-bytes", 0U,
               "max-size-time", 0U, "leaky", 2, nullptr);
  g_object_set(inference_queue, "max-size-buffers", 1U, "max-size-bytes", 0U,
               "max-size-time", 0U, "leaky", 2, nullptr);
  g_object_set(decoder, "cudadec-memtype", 0, "num-extra-surfaces", 4U, nullptr);
  g_object_set(convert, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE, nullptr);
  GstCaps* decoded_rgba = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(decoded_caps, "caps", decoded_rgba, nullptr);
  gst_caps_unref(decoded_rgba);
  GstCaps* inference_rgba = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(inference_convert, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE, nullptr);
  g_object_set(inference_caps, "caps", inference_rgba, nullptr);
  gst_caps_unref(inference_rgba);
  g_object_set(sink, "emit-signals", TRUE, "sync", FALSE, "async", FALSE,
               "max-buffers", 1U, "drop", TRUE, nullptr);
  g_object_set(preview_valve, "drop", context->preview_viewers.load() == 0, nullptr);
  g_object_set(preview_queue, "max-size-buffers", 1U, "max-size-bytes", 0U,
               "max-size-time", 0U, "leaky", 2, nullptr);
  GstCaps* rgba_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(preview_caps, "caps", rgba_caps, nullptr);
  gst_caps_unref(rgba_caps);
  GstCaps* i420_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=I420");
  g_object_set(jpeg_caps, "caps", i420_caps, nullptr);
  gst_caps_unref(i420_caps);
  g_object_set(preview_sink, "sync", FALSE, "async", FALSE, nullptr);
  if (!verify_int_property(decoder, "cudadec-memtype", 0) ||
      !verify_int_property(decoder, "num-extra-surfaces", 4) ||
      !verify_int_property(convert, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE) ||
      !verify_int_property(inference_convert, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE)) {
    context->failures({context->camera, "nvmm_graph_contract", FailureScope::kFatal});
    if (input_caps != nullptr) gst_caps_unref(input_caps);
    return;
  }
  context->preview_valve = preview_valve;
  context->decode_queue = decode_queue;
  g_signal_connect(record_sink, "new-sample", G_CALLBACK(on_encoded_sample), context);
  g_signal_connect(sink, "new-sample", G_CALLBACK(on_inference_sample), context);
  GstPad* jpeg_output = gst_element_get_static_pad(jpeg, "src");
  static_cast<void>(gst_pad_add_probe(jpeg_output, GST_PAD_PROBE_TYPE_BUFFER,
                                      on_preview_encoded, context, nullptr));
  gst_object_unref(jpeg_output);
  gst_bin_add_many(GST_BIN(context->pipeline), depay, parser, parser_caps, tee, record_queue,
                   record_sink, decode_queue, decoder, convert, decoded_caps, transform, decoded_tee,
                   inference_queue, inference_convert, inference_caps, sink, preview_valve,
                   preview_queue, preview_convert, preview_caps, osd, jpeg_convert, jpeg_caps,
                   jpeg, preview_sink, nullptr);
  const bool linked = gst_element_link_many(depay, parser, parser_caps, tee, nullptr) &&
                      gst_element_link_many(tee, record_queue, record_sink, nullptr) &&
                      gst_element_link_many(tee, decode_queue, decoder, convert, decoded_caps,
                                            transform, decoded_tee, nullptr) &&
                      gst_element_link_many(decoded_tee, inference_queue, inference_convert,
                                            inference_caps, sink, nullptr) &&
                      gst_element_link_many(decoded_tee, preview_valve, preview_queue,
                                            preview_convert, preview_caps, osd, jpeg_convert,
                                            jpeg_caps, jpeg, preview_sink, nullptr);
  GstPad* depay_input = gst_element_get_static_pad(depay, "sink");
  const bool source_linked = gst_pad_link(output, depay_input) == GST_PAD_LINK_OK;
  gst_object_unref(depay_input);
  for (GstElement* item : elements) static_cast<void>(gst_element_sync_state_with_parent(item));
  static_cast<void>(gst_element_sync_state_with_parent(preview_sink));
  context->linked = linked && source_linked;
  if (!context->linked) context->failures({context->camera, "parser", FailureScope::kSourceLocal});
  if (input_caps != nullptr) gst_caps_unref(input_caps);
}

void destroy_branch(gpointer raw) { delete static_cast<EncodedSourceContext*>(raw); }
}  // namespace

GstElement* build_encoded_rtsp_pipeline(const std::string& camera, const std::string& uri,
                                         const DeviceFrameCallback& frames,
                                         const FailureCallback& failures,
                                         const AccessUnitCallback& access_units,
                                         const PreviewCallback& previews,
                                         const PipelineBindingPtr& binding,
                                         const std::shared_ptr<InFlightGate>& gate,
                                         std::string* error_code) {
  GstElement* pipeline = gst_pipeline_new(nullptr);
  GstElement* source = element("rtspsrc");
  if (pipeline == nullptr || source == nullptr || !frames || !access_units) {
    *error_code = "encoded_source_unavailable";
    if (source != nullptr) gst_object_unref(source);
    if (pipeline != nullptr) gst_object_unref(pipeline);
    return nullptr;
  }
  g_object_set(source, "location", uri.c_str(), "latency", 200U, nullptr);
  gst_bin_add(GST_BIN(pipeline), source);
  auto* context = new EncodedSourceContext{};
  context->frames = frames;
  context->failures = failures;
  context->access_units = access_units;
  context->previews = previews;
  context->binding = binding;
  context->gate = gate;
  context->camera = camera;
  context->pipeline = pipeline;
  g_object_set_data_full(G_OBJECT(pipeline), "seeon-encoded-context", context, destroy_branch);
  attach_encoded_bus_handler(pipeline, camera, failures);
  g_signal_connect(source, "pad-added", G_CALLBACK(on_rtp_pad), context);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    *error_code = "pipeline_start_failed";
    gst_object_unref(pipeline);
    return nullptr;
  }
  return pipeline;
}

bool set_encoded_preview_viewers(GstElement* pipeline, std::uint32_t viewers) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  if (context == nullptr) return false;
  context->preview_viewers = viewers;
  if (context->preview_valve != nullptr) {
    g_object_set(context->preview_valve, "drop", viewers == 0, nullptr);
  }
  return true;
}

bool wait_encoded_preview(GstElement* pipeline, std::uint64_t target) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  if (context == nullptr) return false;
  std::unique_lock lock{context->preview_mutex};
  return context->preview_ready.wait_for(lock, std::chrono::seconds{2}, [context, target] {
    return context->preview_encoded.load() >= target;
  });
}

void flush_encoded_access_units(GstElement* pipeline) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  if (context != nullptr) flush_pending_access_unit(context);
}

std::optional<PreviewStatus> encoded_preview_status(GstElement* pipeline) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  if (context == nullptr) return std::nullopt;
  return PreviewStatus{context->preview_encoded.load(), context->preview_viewers.load()};
}

bool snapshot_encoded_preview(GstElement* pipeline, std::vector<std::uint8_t>* jpeg) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  if (context == nullptr || context->preview_valve == nullptr) return false;
  const std::uint64_t before = context->preview_encoded.load();
  const bool was_dropping = context->preview_viewers.load() == 0;
  if (was_dropping) g_object_set(context->preview_valve, "drop", FALSE, nullptr);
  bool encoded;
  {
    std::unique_lock lock{context->preview_mutex};
    encoded = context->preview_ready.wait_for(lock, std::chrono::seconds{2},
                                              [context, before] {
                                                return context->preview_encoded.load() > before;
                                              });
    if (encoded) *jpeg = context->last_preview_jpeg;
  }
  if (was_dropping && context->preview_viewers.load() == 0) {
    g_object_set(context->preview_valve, "drop", TRUE, nullptr);
  }
  return encoded && !jpeg->empty();
}

std::uint64_t encoded_au_forwarded(GstElement* pipeline) {
  auto* context = static_cast<EncodedSourceContext*>(
      g_object_get_data(G_OBJECT(pipeline), "seeon-encoded-context"));
  return context == nullptr ? 0 : context->au_forwarded.load();
}
}  // namespace seeon
#endif
