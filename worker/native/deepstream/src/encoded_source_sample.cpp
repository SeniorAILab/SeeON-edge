#include "encoded_source_context.hpp"

#include "au_parser.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <utility>

namespace seeon {
namespace {
std::vector<std::uint8_t> codec_data(const GstStructure* structure) {
  const GValue* value = structure == nullptr ? nullptr : gst_structure_get_value(structure, "codec_data");
  GstBuffer* buffer = value == nullptr ? nullptr : gst_value_get_buffer(value);
  GstMapInfo mapped{};
  if (buffer == nullptr || !gst_buffer_map(buffer, &mapped, GST_MAP_READ)) return {};
  std::vector<std::uint8_t> result{mapped.data, mapped.data + mapped.size};
  gst_buffer_unmap(buffer, &mapped);
  return result;
}
}  // namespace

GstFlowReturn on_encoded_sample(GstAppSink* sink, gpointer raw) {
  auto* context = static_cast<EncodedSourceContext*>(raw);
  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (sample == nullptr) return GST_FLOW_ERROR;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstMapInfo mapped{};
  if (!GST_BUFFER_PTS_IS_VALID(buffer) || !gst_buffer_map(buffer, &mapped, GST_MAP_READ)) {
    context->failures({context->camera, "timestamp", FailureScope::kSourceLocal});
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  GstCaps* caps = gst_sample_get_caps(sample);
  gchar* caps_text = caps == nullptr ? nullptr : gst_caps_to_string(caps);
  const std::string normalized_caps = caps_text == nullptr ? std::string{} : caps_text;
  const bool h265 = normalized_caps.find("video/x-h265") != std::string::npos;
  const bool byte_stream = normalized_caps.find("stream-format=(string)byte-stream") !=
                           std::string::npos;
  const GstStructure* structure = caps == nullptr ? nullptr : gst_caps_get_structure(caps, 0);
  auto configuration = codec_data(structure);
  const std::size_t length_index = h265 ? 21U : 4U;
  const std::size_t length_size = configuration.size() > length_index
                                      ? (configuration[length_index] & 0x03U) + 1U
                                      : 0U;
  const auto annexb = parse_annexb(mapped.data, mapped.size, h265);
  const auto keyframe = byte_stream
                            ? annexb.keyframe
                            : parse_length_prefixed_keyframe(
                                  mapped.data, mapped.size, h265, length_size);
  if (byte_stream && configuration.empty() && !annexb.codec_data.empty()) {
    configuration = annexb.codec_data;
  }
  if (!configuration.empty()) context->codec_data = configuration;
  else configuration = context->codec_data;
  if (!keyframe.has_value() || normalized_caps.find("alignment=(string)au") == std::string::npos) {
    context->failures({context->camera, "keyframe_unknown", FailureScope::kSourceLocal});
  } else {
    gint width = 0;
    gint height = 0;
    gint fps_numerator = 0;
    gint fps_denominator = 1;
    if (structure != nullptr) {
      static_cast<void>(gst_structure_get_int(structure, "width", &width));
      static_cast<void>(gst_structure_get_int(structure, "height", &height));
      static_cast<void>(gst_structure_get_fraction(
          structure, "framerate", &fps_numerator, &fps_denominator));
    }
    const auto pts = static_cast<std::int64_t>(GST_BUFFER_PTS(buffer));
    const auto dts = GST_BUFFER_DTS_IS_VALID(buffer)
                         ? static_cast<std::int64_t>(GST_BUFFER_DTS(buffer)) : pts;
    const auto duration = GST_BUFFER_DURATION_IS_VALID(buffer)
                              ? static_cast<std::int64_t>(GST_BUFFER_DURATION(buffer))
                              : (fps_numerator > 0
                                     ? static_cast<std::int64_t>(GST_SECOND) * fps_denominator /
                                           fps_numerator : 0);
    ParsedAccessUnit unit{h265 ? AuCodec::kH265 : AuCodec::kH264,
                          byte_stream ? AuFraming::kAnnexB : AuFraming::kAvcc,
                          pts, dts, duration, 1, static_cast<std::int32_t>(GST_SECOND),
                          static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                          *keyframe, normalized_caps, configuration,
                          {mapped.data, mapped.data + mapped.size}};
    if (context->pending_duration.has_value()) {
      auto pending = std::move(*context->pending_duration);
      context->pending_duration.reset();
      pending.duration = pts - pending.pts;
      if (pending.duration > 0) context->access_units(context->camera, std::move(pending));
    }
    if (unit.duration > 0) context->access_units(context->camera, std::move(unit));
    else context->pending_duration = std::move(unit);
  }
  g_free(caps_text);
  gst_buffer_unmap(buffer, &mapped);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}
}  // namespace seeon
#endif
