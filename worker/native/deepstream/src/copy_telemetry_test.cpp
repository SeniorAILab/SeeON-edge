#include "copy_telemetry.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sys/resource.h>
#include <sys/stat.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kTelemetryPath = "SEEON_CANARY_TELEMETRY_PATH";

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

class Environment final {
 public:
  Environment() {
    const char* value = std::getenv(kTelemetryPath.data());
    if (value != nullptr) {
      was_set_ = true;
      original_ = value;
    }
  }

  ~Environment() {
    if (!was_set_) {
      (void)unsetenv(kTelemetryPath.data());
    } else {
      (void)setenv(kTelemetryPath.data(), original_.c_str(), 1);
    }
  }

  void unset() const { check(unsetenv(kTelemetryPath.data()) == 0, "could not unset telemetry path"); }

  void set(const std::filesystem::path& path) const {
    check(setenv(kTelemetryPath.data(), path.c_str(), 1) == 0, "could not set telemetry path");
  }

 private:
  bool was_set_ = false;
  std::string original_;
};

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/seeon-copy-telemetry-XXXXXX";
    check(value.size() < pattern.size(), "temporary-directory pattern is too long");
    std::copy(value.begin(), value.end(), pattern.begin());
    check(mkdtemp(pattern.data()) != nullptr, "could not create temporary directory");
    path_ = pattern.data();
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

class FileSizeLimit final {
 public:
  FileSizeLimit() {
    check(getrlimit(RLIMIT_FSIZE, &limit_) == 0, "could not get file-size limit");
    struct sigaction ignored {};
    ignored.sa_handler = SIG_IGN;
    sigemptyset(&ignored.sa_mask);
    check(sigaction(SIGXFSZ, &ignored, &signal_) == 0, "could not ignore file-size signal");
    rlimit restricted = limit_;
    restricted.rlim_cur = 0;
    check(setrlimit(RLIMIT_FSIZE, &restricted) == 0, "could not restrict file size");
  }

  ~FileSizeLimit() {
    (void)setrlimit(RLIMIT_FSIZE, &limit_);
    (void)sigaction(SIGXFSZ, &signal_, nullptr);
  }

 private:
  rlimit limit_{};
  struct sigaction signal_ {};
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::string> lines(const std::string& contents) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start < contents.size()) {
    const std::size_t end = contents.find('\n', start);
    check(end != std::string::npos, "JSONL record is missing newline");
    result.push_back(contents.substr(start, end - start));
    start = end + 1;
  }
  return result;
}

std::uint64_t unsigned_field(const std::string& record, std::string_view field) {
  const std::string prefix = "\"" + std::string(field) + "\":";
  const std::size_t begin = record.find(prefix);
  check(begin != std::string::npos, "missing unsigned field");
  const char* value_begin = record.data() + begin + prefix.size();
  const char* value_end = value_begin;
  while (value_end != record.data() + record.size() && *value_end >= '0' && *value_end <= '9') {
    ++value_end;
  }
  std::uint64_t value = 0;
  const auto [end, error] = std::from_chars(value_begin, value_end, value);
  check(error == std::errc{} && end == value_end, "invalid unsigned field");
  return value;
}

void check_record(const std::string& record, std::string_view camera_id, std::uint64_t frames,
                  std::uint64_t h2d_max, std::uint64_t d2h_max, std::string_view box_source,
                  std::string_view pool_p95, std::string_view gpu_p95, std::uint64_t drops) {
  const std::uint64_t started = unsigned_field(record, "window_started_ns");
  const std::uint64_t ended = unsigned_field(record, "window_ended_ns");
  check(ended >= started, "window end precedes window start");
  const std::string expected =
      "{\"schema_version\":1,\"camera_id\":\"" + std::string(camera_id) +
      "\",\"window_started_ns\":" + std::to_string(started) + ",\"window_ended_ns\":" +
      std::to_string(ended) + ",\"frames\":" + std::to_string(frames) +
      ",\"h2d_bytes_max\":" + std::to_string(h2d_max) + ",\"d2h_bytes_max\":" +
      std::to_string(d2h_max) + ",\"box_source\":\"" + std::string(box_source) +
      "\",\"pool_wait_us_p95\":" + std::string(pool_p95) + ",\"gpu_us_p95\":" +
      std::string(gpu_p95) + ",\"surface_drops\":" + std::to_string(drops) + "}";
  check(record == expected, "JSONL schema, field order, or aggregate values differ");
}

seeon::deepstream::CopyTelemetry::CompletedFrame frame(std::string_view camera_id,
                                                         seeon::deepstream::CopyTelemetry::BoxSource source,
                                                         std::uint64_t h2d, std::uint64_t d2h,
                                                         double pool_wait, double gpu) {
  return {camera_id, h2d, d2h, source, pool_wait, gpu};
}

void configure(seeon::deepstream::CopyTelemetry* telemetry, const std::filesystem::path& parent) {
  std::string error;
  check(seeon::deepstream::CopyTelemetry::from_environment(telemetry, &error),
        "telemetry configuration failed");
  check(telemetry->enabled(), "telemetry was not enabled");
  check(telemetry->path() == parent.parent_path() / (parent.stem().string() + ".child-copy.jsonl"),
        "telemetry did not use the exact sibling sidecar path");
}

}  // namespace

int main() {
  using BoxSource = seeon::deepstream::CopyTelemetry::BoxSource;
  using Telemetry = seeon::deepstream::CopyTelemetry;

  Environment environment;
  TemporaryDirectory temporary_directory;
  const std::filesystem::path parent = temporary_directory.path() / "canary.jsonl";
  const std::filesystem::path sidecar = temporary_directory.path() / "canary.child-copy.jsonl";
  std::ofstream(parent) << "parent record\n";

  environment.unset();
  Telemetry disabled;
  std::string error;
  check(Telemetry::from_environment(&disabled, &error), "absent environment was rejected");
  check(!disabled.enabled(), "absent environment enabled telemetry");
  check(disabled.record_completed_frame(frame("camera", BoxSource::kPose, 1, 1, 1.0, 1.0), &error),
        "disabled telemetry did not no-op");
  check(disabled.flush(&error), "disabled flush did not no-op");
  check(!std::filesystem::exists(sidecar), "disabled telemetry created a sidecar");

  environment.set(parent);
  Telemetry telemetry;
  configure(&telemetry, parent);
  std::ofstream(sidecar).close();
  check(chmod(sidecar.c_str(), 0666) == 0, "could not set runner sidecar permissions");
  check(telemetry.record_completed_frame(frame("camera", BoxSource::kPose, 7, 9, 1.0, 2.0), &error),
        "first pose frame was rejected");
  check(telemetry.record_completed_frame(frame("camera", BoxSource::kPose, 11, 13, 5.0, 10.0), &error),
        "second pose frame was rejected");
  check(telemetry.record_busy_surface_drop("camera", BoxSource::kPose, &error),
        "pose surface drop was rejected");
  check(telemetry.record_completed_frame(frame("camera", BoxSource::kPerson, 3, 4, 7.0, 8.0), &error),
        "person frame was rejected");
  check(telemetry.record_busy_surface_drop("camera", BoxSource::kPerson, &error),
        "person surface drop was rejected");
  check(telemetry.flush(&error), "flush failed");
  check(read_file(parent) == "parent record\n", "flush modified the configured parent file");
  struct stat sidecar_status {};
  check(stat(sidecar.c_str(), &sidecar_status) == 0 && (sidecar_status.st_mode & 0777) == 0666,
        "flush changed runner sidecar permissions");
  const std::vector<std::string> records = lines(read_file(sidecar));
  check(records.size() == 2, "flush did not write both source windows");
  check_record(records[0], "camera", 2, 11, 13, "pose", "4.8", "9.6", 1);
  check_record(records[1], "camera", 1, 3, 4, "person", "7", "8", 1);
  check(telemetry.flush(&error), "second flush failed");
  check(lines(read_file(sidecar)).size() == 2, "flush duplicated sealed records");

  Telemetry invalid;
  configure(&invalid, parent);
  const std::string invalid_camera(1, static_cast<char>(0xff));
  check(!invalid.record_busy_surface_drop(invalid_camera, BoxSource::kPose, &error) &&
            error == "copy telemetry: invalid field",
        "invalid camera did not return the stable invalid-field error");
  check(!invalid.record_completed_frame(
            frame("camera", BoxSource::kPose, 1, 1, std::numeric_limits<double>::quiet_NaN(), 1.0),
            &error) &&
            error == "copy telemetry: invalid field",
        "nonfinite pool metric did not return the stable invalid-field error");
  check(!invalid.record_completed_frame(
            frame("camera", BoxSource::kPose, 1, 1, 1.0, std::numeric_limits<double>::infinity()),
            &error) &&
            error == "copy telemetry: invalid field",
        "nonfinite GPU metric did not return the stable invalid-field error");

  const std::filesystem::path missing_parent = temporary_directory.path() / "missing" / "canary.jsonl";
  environment.set(missing_parent);
  Telemetry bad_parent;
  configure(&bad_parent, missing_parent);
  check(bad_parent.record_busy_surface_drop("camera", BoxSource::kPose, &error),
        "bad-parent window was rejected before flush");
  check(!bad_parent.flush(&error) && error == "copy telemetry: open failed",
        "bad parent did not return the stable open error");

  const std::filesystem::path write_parent = temporary_directory.path() / "write.jsonl";
  const std::filesystem::path write_sidecar = temporary_directory.path() / "write.child-copy.jsonl";
  std::filesystem::create_symlink("/dev/full", write_sidecar);
  environment.set(write_parent);
  Telemetry write_failure;
  configure(&write_failure, write_parent);
  check(write_failure.record_busy_surface_drop("camera", BoxSource::kPose, &error),
        "write-failure window was rejected before flush");
  check(!write_failure.flush(&error) && error == "copy telemetry: open failed",
        "symlink sidecar did not return the stable open error");

  const std::filesystem::path limited_parent = temporary_directory.path() / "limited.jsonl";
  const std::filesystem::path limited_sidecar =
      temporary_directory.path() / "limited.child-copy.jsonl";
  std::ofstream(limited_sidecar).close();
  environment.set(limited_parent);
  Telemetry limited_write_failure;
  configure(&limited_write_failure, limited_parent);
  check(limited_write_failure.record_busy_surface_drop("camera", BoxSource::kPose, &error),
        "limited-write window was rejected before flush");
  {
    FileSizeLimit limit;
    check(!limited_write_failure.flush(&error) && error == "copy telemetry: write failed",
          "regular-file write failure did not return the stable write error");
  }
  return 0;
}
