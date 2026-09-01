#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace seeon::deepstream {

class CopyTelemetry final {
 public:
  static_assert(std::atomic<bool>::is_always_lock_free);

  enum class BoxSource : std::uint8_t {
    kPose,
    kPerson,
  };

  struct CompletedFrame {
    std::string_view camera_id;
    std::uint64_t h2d_bytes;
    std::uint64_t d2h_bytes;
    BoxSource box_source;
    double pool_wait_us;
    double gpu_us;
  };

  CopyTelemetry() = default;
  CopyTelemetry(const CopyTelemetry&) = delete;
  CopyTelemetry& operator=(const CopyTelemetry&) = delete;

  // Configures an opt-in sidecar from SEEON_CANARY_TELEMETRY_PATH. An absent
  // or empty variable leaves telemetry disabled and all record calls are no-ops.
  static bool from_environment(CopyTelemetry* telemetry, std::string* error);

  [[nodiscard]] bool enabled() const;
  [[nodiscard]] std::string path() const;

  // Records a completed inference frame. box_source is retained in the emitted
  // record, so drops are attributed with the same source by record_busy_surface_drop.
  bool record_completed_frame(const CompletedFrame& frame, std::string* error);
  bool record_busy_surface_drop(
      std::string_view camera_id, BoxSource box_source, std::string* error);

  // Seals all active windows. Disabled telemetry is a no-op.
  bool flush(std::string* error);

 private:
  struct Window {
    bool active = false;
    std::uint64_t started_ns = 0;
    std::uint64_t steady_started_ns = 0;
    std::uint64_t frames = 0;
    std::uint64_t h2d_bytes_max = 0;
    std::uint64_t d2h_bytes_max = 0;
    std::uint64_t surface_drops = 0;
    std::vector<double> pool_wait_us;
    std::vector<double> gpu_us;
  };

  struct CameraWindows {
    Window pose;
    Window person;
  };

  static Window& window_for(CameraWindows& windows, BoxSource box_source);
  static bool flush_window(const std::string& path, std::string_view camera_id,
                           BoxSource box_source, Window* window, std::uint64_t ended_ns,
                           std::string* error);
  static bool elapsed_window(const Window& window, std::uint64_t steady_now_ns);
  static void start_window(
      Window* window, std::uint64_t realtime_now_ns, std::uint64_t steady_now_ns);

  std::atomic<bool> configured_{false};
  std::string path_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, CameraWindows> cameras_;

  bool record_completed_frame_locked(const CompletedFrame& frame, std::string* error);
  bool record_busy_surface_drop_locked(
      std::string_view camera_id, BoxSource box_source, std::string* error);
};

}  // namespace seeon::deepstream
