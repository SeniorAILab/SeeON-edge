#include "copy_telemetry.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

namespace seeon::deepstream {
namespace {

constexpr std::uint64_t kWindowNanoseconds = 10'000'000'000ULL;
constexpr const char* kInvalidPath = "copy telemetry: invalid path";
constexpr const char* kInvalidField = "copy telemetry: invalid field";
constexpr const char* kOpenFailed = "copy telemetry: open failed";
constexpr const char* kWriteFailed = "copy telemetry: write failed";
constexpr const char* kFsyncFailed = "copy telemetry: fsync failed";
constexpr const char* kCloseFailed = "copy telemetry: close failed";
constexpr const char* kInternalFailed = "copy telemetry: internal failure";

void set_error(std::string* error, const char* value) {
  if (error != nullptr) {
    *error = value;
  }
}

bool valid_utf8(std::string_view value) {
  for (std::size_t index = 0; index < value.size();) {
    const auto lead = static_cast<unsigned char>(value[index]);
    if (lead <= 0x7fU) {
      if (lead < 0x20U) {
        return false;
      }
      ++index;
      continue;
    }

    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (lead >= 0xc2U && lead <= 0xdfU) {
      continuation_count = 1;
      code_point = lead & 0x1fU;
    } else if (lead >= 0xe0U && lead <= 0xefU) {
      continuation_count = 2;
      code_point = lead & 0x0fU;
    } else if (lead >= 0xf0U && lead <= 0xf4U) {
      continuation_count = 3;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (index + continuation_count >= value.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto byte = static_cast<unsigned char>(value[index + offset]);
      if ((byte & 0xc0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (byte & 0x3fU);
    }
    if ((continuation_count == 2 && code_point < 0x800U) ||
        (continuation_count == 3 && code_point < 0x10000U) ||
        (code_point >= 0xd800U && code_point <= 0xdfffU) || code_point > 0x10ffffU) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

bool valid_camera_id(std::string_view camera_id) {
  return !camera_id.empty() && camera_id.find('\0') == std::string_view::npos && valid_utf8(camera_id);
}

const char* box_source_name(CopyTelemetry::BoxSource box_source) {
  switch (box_source) {
    case CopyTelemetry::BoxSource::kPose:
      return "pose";
    case CopyTelemetry::BoxSource::kPerson:
      return "person";
  }
  return nullptr;
}

std::uint64_t realtime_nanoseconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

std::uint64_t steady_nanoseconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

double percentile95(std::vector<double> samples) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  const double position = static_cast<double>(samples.size() - 1) * 0.95;
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, samples.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return samples[lower] * (1.0 - weight) + samples[upper] * weight;
}

void append_json_string(std::string* target, std::string_view value) {
  target->push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        target->append("\\\"");
        break;
      case '\\':
        target->append("\\\\");
        break;
      case '\b':
        target->append("\\b");
        break;
      case '\f':
        target->append("\\f");
        break;
      case '\n':
        target->append("\\n");
        break;
      case '\r':
        target->append("\\r");
        break;
      case '\t':
        target->append("\\t");
        break;
      default:
        target->push_back(static_cast<char>(byte));
        break;
    }
  }
  target->push_back('"');
}

template <typename Number>
void append_number(std::string* target, Number value) {
  std::array<char, 64> buffer{};
  const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error == std::errc{}) {
    target->append(buffer.data(), end);
  }
}

bool write_record(const std::string& path, const std::string& record, std::string* error) {
  const int descriptor =
      open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK,
           0600);
  if (descriptor < 0) {
    set_error(error, kOpenFailed);
    return false;
  }
  struct stat status {};
  if (fstat(descriptor, &status) < 0 || !S_ISREG(status.st_mode)) {
    (void)close(descriptor);
    set_error(error, kOpenFailed);
    return false;
  }

  std::size_t written = 0;
  while (written < record.size()) {
    const ssize_t result = write(descriptor, record.data() + written, record.size() - written);
    if (result > 0) {
      written += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    (void)close(descriptor);
    set_error(error, kWriteFailed);
    return false;
  }

  int sync_result = 0;
  do {
    sync_result = fsync(descriptor);
  } while (sync_result < 0 && errno == EINTR);
  if (sync_result < 0) {
    (void)close(descriptor);
    set_error(error, kFsyncFailed);
    return false;
  }
  if (close(descriptor) < 0) {
    set_error(error, kCloseFailed);
    return false;
  }
  return true;
}

}  // namespace

CopyTelemetry::Window& CopyTelemetry::window_for(
    CameraWindows& windows, BoxSource box_source) {
  return box_source == BoxSource::kPose ? windows.pose : windows.person;
}

bool CopyTelemetry::flush_window(const std::string& path, std::string_view camera_id,
                                 BoxSource box_source, Window* window,
                                 std::uint64_t ended_ns, std::string* error) {
  if (!window->active) {
    return true;
  }
  ended_ns = std::max(ended_ns, window->started_ns);
  std::string record;
  record.reserve(256 + camera_id.size());
  record.append("{\"schema_version\":1,\"camera_id\":");
  append_json_string(&record, camera_id);
  record.append(",\"window_started_ns\":");
  append_number(&record, window->started_ns);
  record.append(",\"window_ended_ns\":");
  append_number(&record, ended_ns);
  record.append(",\"frames\":");
  append_number(&record, window->frames);
  record.append(",\"h2d_bytes_max\":");
  append_number(&record, window->h2d_bytes_max);
  record.append(",\"d2h_bytes_max\":");
  append_number(&record, window->d2h_bytes_max);
  record.append(",\"box_source\":");
  append_json_string(&record, box_source_name(box_source));
  record.append(",\"pool_wait_us_p95\":");
  append_number(&record, percentile95(window->pool_wait_us));
  record.append(",\"gpu_us_p95\":");
  append_number(&record, percentile95(window->gpu_us));
  record.append(",\"surface_drops\":");
  append_number(&record, window->surface_drops);
  record.append("}\n");

  if (!write_record(path, record, error)) {
    return false;
  }
  *window = Window{};
  return true;
}

bool CopyTelemetry::elapsed_window(const Window& window, std::uint64_t steady_now_ns) {
  return window.active && steady_now_ns >= window.steady_started_ns &&
         steady_now_ns - window.steady_started_ns >= kWindowNanoseconds;
}

void CopyTelemetry::start_window(Window* window, std::uint64_t realtime_now_ns,
                                 std::uint64_t steady_now_ns) {
  window->active = true;
  window->started_ns = realtime_now_ns;
  window->steady_started_ns = steady_now_ns;
}

void increment_saturating(std::uint64_t* value) {
  if (*value != std::numeric_limits<std::uint64_t>::max()) {
    ++*value;
  }
}

bool CopyTelemetry::from_environment(CopyTelemetry* telemetry, std::string* error) {
  if (telemetry == nullptr) {
    set_error(error, kInvalidField);
    return false;
  }
  telemetry->configured_.store(false, std::memory_order_release);
  {
    std::lock_guard lock(telemetry->mutex_);
    telemetry->path_.clear();
    telemetry->cameras_.clear();
  }
  const char* configured_path = std::getenv("SEEON_CANARY_TELEMETRY_PATH");
  if (configured_path == nullptr || configured_path[0] == '\0') {
    return true;
  }

  try {
    const std::string raw_path(configured_path);
    if (raw_path.find('\0') != std::string::npos) {
      set_error(error, kInvalidPath);
      return false;
    }
    const std::filesystem::path parent_path(raw_path);
    const std::filesystem::path filename = parent_path.filename();
    const std::string stem = filename.stem().string();
    if (filename.empty() || filename == "." || filename == ".." || stem.empty()) {
      set_error(error, kInvalidPath);
      return false;
    }
    const std::filesystem::path sidecar =
        parent_path.parent_path() / (stem + ".child-copy.jsonl");
    const std::string sidecar_path = sidecar.string();
    if (sidecar_path.empty() || sidecar_path.find('\0') != std::string::npos) {
      set_error(error, kInvalidPath);
      return false;
    }
    std::lock_guard lock(telemetry->mutex_);
    telemetry->cameras_.clear();
    telemetry->path_ = sidecar_path;
    telemetry->configured_.store(true, std::memory_order_release);
    return true;
  } catch (const std::exception&) {
    set_error(error, kInvalidPath);
    return false;
  }
}

bool CopyTelemetry::enabled() const {
  return configured_.load(std::memory_order_acquire);
}

std::string CopyTelemetry::path() const {
  std::lock_guard lock(mutex_);
  return path_;
}

bool CopyTelemetry::record_completed_frame(const CompletedFrame& frame, std::string* error) {
  if (!configured_.load(std::memory_order_acquire)) {
    return true;
  }
  try {
    std::lock_guard lock(mutex_);
    return record_completed_frame_locked(frame, error);
  } catch (const std::exception&) {
    set_error(error, kInternalFailed);
    return false;
  }
}

bool CopyTelemetry::record_busy_surface_drop(
    std::string_view camera_id, BoxSource box_source, std::string* error) {
  if (!configured_.load(std::memory_order_acquire)) {
    return true;
  }
  try {
    std::lock_guard lock(mutex_);
    return record_busy_surface_drop_locked(camera_id, box_source, error);
  } catch (const std::exception&) {
    set_error(error, kInternalFailed);
    return false;
  }
}

bool CopyTelemetry::flush(std::string* error) {
  if (!configured_.load(std::memory_order_acquire)) {
    return true;
  }
  try {
    std::lock_guard lock(mutex_);
    const std::uint64_t ended_ns = realtime_nanoseconds();
    for (auto& [camera_id, windows] : cameras_) {
      if (!flush_window(path_, camera_id, BoxSource::kPose, &windows.pose, ended_ns, error) ||
          !flush_window(path_, camera_id, BoxSource::kPerson, &windows.person, ended_ns, error)) {
        return false;
      }
    }
    return true;
  } catch (const std::exception&) {
    set_error(error, kInternalFailed);
    return false;
  }
}

bool CopyTelemetry::record_completed_frame_locked(const CompletedFrame& frame, std::string* error) {
  if (!valid_camera_id(frame.camera_id) || box_source_name(frame.box_source) == nullptr ||
      !std::isfinite(frame.pool_wait_us) || std::signbit(frame.pool_wait_us) ||
      !std::isfinite(frame.gpu_us) || std::signbit(frame.gpu_us)) {
    set_error(error, kInvalidField);
    return false;
  }

  const std::uint64_t realtime_now_ns = realtime_nanoseconds();
  const std::uint64_t steady_now_ns = steady_nanoseconds();
  auto& window = window_for(cameras_[std::string(frame.camera_id)], frame.box_source);
  if (elapsed_window(window, steady_now_ns) &&
      !flush_window(path_, frame.camera_id, frame.box_source, &window, realtime_now_ns, error)) {
    return false;
  }
  if (!window.active) {
    start_window(&window, realtime_now_ns, steady_now_ns);
  }
  increment_saturating(&window.frames);
  window.h2d_bytes_max = std::max(window.h2d_bytes_max, frame.h2d_bytes);
  window.d2h_bytes_max = std::max(window.d2h_bytes_max, frame.d2h_bytes);
  window.pool_wait_us.push_back(frame.pool_wait_us);
  window.gpu_us.push_back(frame.gpu_us);
  return true;
}

bool CopyTelemetry::record_busy_surface_drop_locked(
    std::string_view camera_id, BoxSource box_source, std::string* error) {
  if (!valid_camera_id(camera_id) || box_source_name(box_source) == nullptr) {
    set_error(error, kInvalidField);
    return false;
  }

  const std::uint64_t realtime_now_ns = realtime_nanoseconds();
  const std::uint64_t steady_now_ns = steady_nanoseconds();
  auto& window = window_for(cameras_[std::string(camera_id)], box_source);
  if (elapsed_window(window, steady_now_ns) &&
      !flush_window(path_, camera_id, box_source, &window, realtime_now_ns, error)) {
    return false;
  }
  if (!window.active) {
    start_window(&window, realtime_now_ns, steady_now_ns);
  }
  increment_saturating(&window.surface_drops);
  return true;
}

}  // namespace seeon::deepstream
