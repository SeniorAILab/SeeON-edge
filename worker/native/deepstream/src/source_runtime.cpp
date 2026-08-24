#include "source_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <utility>

#ifdef SEEON_HAS_GSTREAMER
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

namespace {
typedef struct _SeeonPerceptionTransform {
  GstBaseTransform parent;
} SeeonPerceptionTransform;
typedef struct _SeeonPerceptionTransformClass {
  GstBaseTransformClass parent_class;
} SeeonPerceptionTransformClass;
G_DEFINE_TYPE(SeeonPerceptionTransform, seeon_perception_transform, GST_TYPE_BASE_TRANSFORM)

GstFlowReturn transform_in_place(GstBaseTransform*, GstBuffer*) { return GST_FLOW_OK; }
void seeon_perception_transform_class_init(SeeonPerceptionTransformClass* klass) {
  auto* element = GST_ELEMENT_CLASS(klass);
  auto* transform = GST_BASE_TRANSFORM_CLASS(klass);
  gst_element_class_set_static_metadata(element, "SeeON perception transform", "Filter/Metadata",
                                        "Attaches compact SeeON perception metadata", "SeniorAILab");
  GstCaps* caps = gst_caps_new_any();
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
  transform->transform_ip = transform_in_place;
  transform->passthrough_on_same_caps = true;
}
void seeon_perception_transform_init(SeeonPerceptionTransform* transform) {
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(transform), true);
}

struct SourceContext {
  seeon::FrameCallback frame_callback;
  seeon::FailureCallback failure_callback;
  std::string camera;
};

GstPadProbeReturn on_buffer(GstPad*, GstPadProbeInfo* info, gpointer raw) {
  auto* context = static_cast<SourceContext*>(raw);
  GstBuffer* buffer = gst_pad_probe_info_get_buffer(info);
  context->frame_callback(context->camera,
                          GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0);
  return GST_PAD_PROBE_OK;
}
void destroy_context(gpointer raw) { delete static_cast<SourceContext*>(raw); }

void on_decode_pad(GstElement*, GstPad* output, gpointer raw_convert) {
  GstPad* input = gst_element_get_static_pad(GST_ELEMENT(raw_convert), "sink");
  if (!gst_pad_is_linked(input)) {
    static_cast<void>(gst_pad_link(output, input));
  }
  gst_object_unref(input);
}

bool contains(const std::string& value, const std::string& token) {
  return value.find(token) != std::string::npos;
}

bool contains_fatal_token(std::string value) {
  std::ranges::transform(value, value.begin(),
                         [](unsigned char character) { return std::tolower(character); });
  return contains(value, "cuda") || contains(value, "xid") || contains(value, "context") ||
         contains(value, "tensorrt") || contains(value, "out of memory") ||
         contains(value, "glib");
}

GstBusSyncReply on_bus(GstBus*, GstMessage* message, gpointer raw) {
  auto* context = static_cast<SourceContext*>(raw);
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
    context->failure_callback({context->camera, "eos", seeon::FailureScope::kSourceLocal});
    return GST_BUS_DROP;
  }
  if (GST_MESSAGE_TYPE(message) != GST_MESSAGE_ERROR) {
    return GST_BUS_PASS;
  }
  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  const std::string stable = error == nullptr ? "native_error" : error->message;
  const bool fatal =
      contains_fatal_token(stable) ||
      contains(std::string{GST_OBJECT_NAME(GST_MESSAGE_SRC(message))}, "perception");
  context->failure_callback({context->camera, fatal ? "shared_pipeline" : "decoder_source",
                             fatal ? seeon::FailureScope::kFatal
                                   : seeon::FailureScope::kSourceLocal});
  g_clear_error(&error);
  g_free(debug);
  return GST_BUS_DROP;
}
}  // namespace
#endif

namespace seeon {
class SourceRuntime::Impl {
 public:
#ifdef SEEON_HAS_GSTREAMER
  struct Source { std::string uri; GstElement* pipeline; };
  std::map<std::string, Source> sources;
  bool transform_available = false;
#else
  std::map<std::string, std::string> sources;
  bool transform_available = true;
#endif
};

SourceRuntime::SourceRuntime(FrameCallback frame_callback, FailureCallback failure_callback)
    : frame_callback_(std::move(frame_callback)),
      failure_callback_(std::move(failure_callback)),
      impl_(std::make_unique<Impl>()) {
#ifdef SEEON_HAS_GSTREAMER
  gst_init(nullptr, nullptr);
  impl_->transform_available = gst_element_register(
      nullptr, "seeonperceptiontransform", GST_RANK_NONE, seeon_perception_transform_get_type());
#endif
}

SourceRuntime::~SourceRuntime() {
#ifdef SEEON_HAS_GSTREAMER
  for (auto& [camera, source] : impl_->sources) {
    static_cast<void>(camera);
    gst_element_set_state(source.pipeline, GST_STATE_NULL);
    gst_object_unref(source.pipeline);
  }
#endif
}

#ifdef SEEON_HAS_GSTREAMER
GstElement* build_pipeline(const std::string& camera, const std::string& uri,
                           const FrameCallback& frames, const FailureCallback& failures,
                           std::string* error_code) {
  GstElement* pipeline = gst_pipeline_new(nullptr);
  GstElement* source = gst_element_factory_make(uri.starts_with("loopback://")
                                                    ? "videotestsrc" : "uridecodebin", nullptr);
  GstElement* convert = gst_element_factory_make("videoconvert", nullptr);
  GstElement* rgba = gst_element_factory_make("capsfilter", nullptr);
  GstElement* nvconvert = gst_element_factory_make("nvvideoconvert", nullptr);
  GstElement* nvmm = gst_element_factory_make("capsfilter", nullptr);
  GstElement* transform = gst_element_factory_make("seeonperceptiontransform", nullptr);
  GstElement* sink = gst_element_factory_make("fakesink", nullptr);
  if (pipeline == nullptr || source == nullptr || convert == nullptr || rgba == nullptr ||
      nvconvert == nullptr || nvmm == nullptr || transform == nullptr || sink == nullptr) {
    *error_code = "element_unavailable";
    if (pipeline != nullptr) gst_object_unref(pipeline);
    return nullptr;
  }
  if (uri.starts_with("loopback://")) {
    g_object_set(source, "is-live", TRUE, "pattern", 2, nullptr);
  } else {
    g_object_set(source, "uri", uri.c_str(), nullptr);
    g_signal_connect(source, "pad-added", G_CALLBACK(on_decode_pad), convert);
  }
  GstCaps* rgba_caps = gst_caps_from_string("video/x-raw,format=RGBA");
  GstCaps* nvmm_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
  g_object_set(rgba, "caps", rgba_caps, nullptr);
  g_object_set(nvmm, "caps", nvmm_caps, nullptr);
  g_object_set(sink, "sync", FALSE, nullptr);
  gst_caps_unref(rgba_caps);
  gst_caps_unref(nvmm_caps);
  gst_bin_add_many(GST_BIN(pipeline), source, convert, rgba, nvconvert, nvmm, transform, sink, nullptr);
  const bool downstream = gst_element_link_many(convert, rgba, nvconvert, nvmm, transform, sink, nullptr);
  const bool upstream = !uri.starts_with("loopback://") || gst_element_link(source, convert);
  if (!downstream || !upstream) {
    *error_code = "element_link_failed";
    gst_object_unref(pipeline);
    return nullptr;
  }
  auto* context = new SourceContext{frames, failures, camera};
  GstPad* output = gst_element_get_static_pad(transform, "src");
  static_cast<void>(gst_pad_add_probe(output, GST_PAD_PROBE_TYPE_BUFFER, on_buffer, context,
                                      destroy_context));
  gst_object_unref(output);
  GstBus* bus = gst_element_get_bus(pipeline);
  gst_bus_set_sync_handler(bus, on_bus, new SourceContext{frames, failures, camera}, destroy_context);
  gst_object_unref(bus);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    *error_code = "pipeline_start_failed";
    gst_object_unref(pipeline);
    return nullptr;
  }
  return pipeline;
}
#endif

bool SourceRuntime::add(const std::string& camera, const std::string& uri, std::string* error_code) {
  if (!valid_source_uri(uri)) { *error_code = "source_uri_invalid"; return false; }
  if (impl_->sources.contains(camera)) { *error_code = "source_exists"; return false; }
#ifdef SEEON_HAS_GSTREAMER
  GstElement* pipeline = build_pipeline(camera, uri, frame_callback_, failure_callback_, error_code);
  if (pipeline == nullptr) return false;
  impl_->sources.emplace(camera, Impl::Source{uri, pipeline});
#else
  impl_->sources.emplace(camera, uri);
  frame_callback_(camera, 0);
#endif
  return true;
}

bool SourceRuntime::remove(const std::string& camera) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) return false;
#ifdef SEEON_HAS_GSTREAMER
  gst_element_set_state(found->second.pipeline, GST_STATE_NULL);
  gst_object_unref(found->second.pipeline);
#endif
  impl_->sources.erase(found);
  return true;
}

bool SourceRuntime::rebuild(const std::string& camera, std::string* error_code) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) { *error_code = "source_unknown"; return false; }
#ifdef SEEON_HAS_GSTREAMER
  GstElement* replacement = build_pipeline(camera, found->second.uri, frame_callback_,
                                            failure_callback_, error_code);
  if (replacement == nullptr) return false;
  gst_element_set_state(found->second.pipeline, GST_STATE_NULL);
  gst_object_unref(found->second.pipeline);
  found->second.pipeline = replacement;
#else
  frame_callback_(camera, 0);
#endif
  return true;
}

bool SourceRuntime::inject_eos(const std::string& camera) {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found != impl_->sources.end() &&
         gst_element_send_event(found->second.pipeline, gst_event_new_eos());
#else
  failure_callback_({camera, "eos", FailureScope::kSourceLocal});
  return impl_->sources.contains(camera);
#endif
}
std::size_t SourceRuntime::count() const { return impl_->sources.size(); }
bool SourceRuntime::custom_transform_available() const { return impl_->transform_available; }
}  // namespace seeon
