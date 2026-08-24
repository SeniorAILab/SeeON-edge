#include "source_runtime.hpp"

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

GstFlowReturn transform_in_place(GstBaseTransform* transform, GstBuffer* buffer) {
  static_cast<void>(transform);
  static_cast<void>(buffer);
  return GST_FLOW_OK;
}

void seeon_perception_transform_class_init(SeeonPerceptionTransformClass* klass) {
  GstElementClass* element = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass* transform = GST_BASE_TRANSFORM_CLASS(klass);
  gst_element_class_set_static_metadata(
      element,
      "SeeON perception transform",
      "Filter/Metadata",
      "Attaches compact SeeON perception metadata",
      "SeniorAILab");
  GstCaps* caps = gst_caps_new_any();
  gst_element_class_add_pad_template(
      element,
      gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template(
      element,
      gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
  transform->transform_ip = transform_in_place;
  transform->passthrough_on_same_caps = true;
}

void seeon_perception_transform_init(SeeonPerceptionTransform* transform) {
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(transform), true);
}

struct ProbeContext {
  seeon::FrameCallback callback;
  std::string camera;
};

GstPadProbeReturn on_transform_buffer(GstPad* pad, GstPadProbeInfo* info, gpointer raw_context) {
  static_cast<void>(pad);
  auto* context = static_cast<ProbeContext*>(raw_context);
  GstBuffer* buffer = gst_pad_probe_info_get_buffer(info);
  const auto pts = GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0;
  context->callback(context->camera, pts);
  return GST_PAD_PROBE_OK;
}

void destroy_probe_context(gpointer raw_context) {
  delete static_cast<ProbeContext*>(raw_context);
}
}  // namespace
#endif

namespace seeon {
class SourceRuntime::Impl {
 public:
#ifdef SEEON_HAS_GSTREAMER
  struct Source {
    std::string uri;
    GstElement* pipeline;
  };
  std::map<std::string, Source> sources;
  bool transform_available = false;
#else
  std::map<std::string, std::string> sources;
  bool transform_available = true;
#endif
};

SourceRuntime::SourceRuntime(FrameCallback callback)
    : callback_(std::move(callback)), impl_(std::make_unique<Impl>()) {
#ifdef SEEON_HAS_GSTREAMER
  gst_init(nullptr, nullptr);
  impl_->transform_available = gst_element_register(
      nullptr,
      "seeonperceptiontransform",
      GST_RANK_NONE,
      seeon_perception_transform_get_type());
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

bool SourceRuntime::add(const std::string& camera, const std::string& uri, std::string* error) {
  if (impl_->sources.contains(camera)) {
    *error = "source already exists";
    return false;
  }
#ifdef SEEON_HAS_GSTREAMER
  if (!impl_->transform_available) {
    *error = "custom transform unavailable";
    return false;
  }
  gchar* quoted = g_shell_quote(uri.c_str());
  const std::string source = uri.starts_with("loopback://")
                                 ? "videotestsrc is-live=true pattern=black"
                                 : "uridecodebin uri=" + std::string{quoted};
  g_free(quoted);
  const std::string description =
      source + " ! videoconvert ! video/x-raw,format=RGBA ! nvvideoconvert ! "
               "video/x-raw(memory:NVMM),format=NV12 ! "
               "seeonperceptiontransform name=perception ! "
               "fakesink sync=false";
  GError* parse_error = nullptr;
  GstElement* pipeline = gst_parse_launch(description.c_str(), &parse_error);
  if (pipeline == nullptr || parse_error != nullptr) {
    *error = parse_error == nullptr ? "pipeline parse failed" : parse_error->message;
    g_clear_error(&parse_error);
    if (pipeline != nullptr) {
      gst_object_unref(pipeline);
    }
    return false;
  }
  GstElement* transform = gst_bin_get_by_name(GST_BIN(pipeline), "perception");
  GstPad* output = transform == nullptr ? nullptr : gst_element_get_static_pad(transform, "src");
  if (output == nullptr) {
    if (transform != nullptr) {
      gst_object_unref(transform);
    }
    gst_object_unref(pipeline);
    *error = "custom transform output unavailable";
    return false;
  }
  gst_pad_add_probe(
      output,
      GST_PAD_PROBE_TYPE_BUFFER,
      on_transform_buffer,
      new ProbeContext{callback_, camera},
      destroy_probe_context);
  gst_object_unref(output);
  gst_object_unref(transform);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    *error = "pipeline refused PLAYING state";
    return false;
  }
  GstState current = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  const auto settled = gst_element_get_state(pipeline, &current, &pending, 5 * GST_SECOND);
  if (settled == GST_STATE_CHANGE_FAILURE || settled == GST_STATE_CHANGE_ASYNC) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    *error = "pipeline did not establish NVMM path";
    return false;
  }
  impl_->sources.emplace(camera, Impl::Source{uri, pipeline});
#else
  static_cast<void>(error);
  impl_->sources.emplace(camera, uri);
  callback_(camera, 0);
#endif
  return true;
}

bool SourceRuntime::remove(const std::string& camera) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) {
    return false;
  }
#ifdef SEEON_HAS_GSTREAMER
  gst_element_set_state(found->second.pipeline, GST_STATE_NULL);
  gst_object_unref(found->second.pipeline);
#endif
  impl_->sources.erase(found);
  return true;
}

bool SourceRuntime::rebuild(const std::string& camera, std::string* error) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) {
    *error = "unknown source";
    return false;
  }
#ifdef SEEON_HAS_GSTREAMER
  const std::string uri = found->second.uri;
#else
  const std::string uri = found->second;
#endif
  static_cast<void>(remove(camera));
  return add(camera, uri, error);
}

std::size_t SourceRuntime::count() const { return impl_->sources.size(); }

bool SourceRuntime::custom_transform_available() const { return impl_->transform_available; }
}  // namespace seeon
