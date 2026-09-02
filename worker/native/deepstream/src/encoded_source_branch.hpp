#pragma once

#include "source_runtime.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/gst.h>

#include <memory>
#include <string>

namespace seeon {
struct InFlightGate;
[[nodiscard]] GstElement* build_encoded_rtsp_pipeline(const std::string& camera,
                                                       const std::string& uri,
                                                       const DeviceFrameCallback& frames,
                                                       const FailureCallback& failures,
                                                       const AccessUnitCallback& access_units,
                                                       const PreviewCallback& previews,
                                                       const PipelineBindingPtr& binding,
                                                       const std::shared_ptr<InFlightGate>& gate,
                                                       std::string* error_code);
[[nodiscard]] bool set_encoded_preview_viewers(GstElement* pipeline, std::uint32_t viewers);
[[nodiscard]] std::optional<PreviewStatus> encoded_preview_status(GstElement* pipeline);
[[nodiscard]] bool wait_encoded_preview(GstElement* pipeline, std::uint64_t target);
[[nodiscard]] bool snapshot_encoded_preview(GstElement* pipeline,
                                            std::vector<std::uint8_t>* jpeg);
[[nodiscard]] std::uint64_t encoded_au_forwarded(GstElement* pipeline);
void flush_encoded_access_units(GstElement* pipeline);
}  // namespace seeon
#endif
