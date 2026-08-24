#include "source_runtime.hpp"

#include <cassert>
#include <string>
#include <vector>

int main() {
  assert(seeon::valid_source_uri("loopback://camera-a"));
  assert(seeon::valid_source_uri("rtsp://user:p'ass@camera.example/live"));
  assert(!seeon::valid_source_uri("file:///tmp/input"));
  assert(!seeon::valid_source_uri("http://camera.example/live"));
  assert(!seeon::valid_source_uri("rtsp://camera.example/live\nfilesink"));

  std::vector<std::string> frames;
  std::vector<seeon::NativeFailure> failures;
  seeon::SourceRuntime runtime(
      [&frames](const std::string& camera, std::uint64_t) { frames.push_back(camera); },
      [&failures](const seeon::NativeFailure& failure) { failures.push_back(failure); });
  std::string error_code;
  assert(!runtime.add("camera-file", "file:///tmp/input", &error_code));
  assert(error_code == "source_uri_invalid");
  assert(runtime.add("camera-a", "loopback://camera-a", &error_code));
  assert(runtime.count() == 1);
  assert(!frames.empty() && frames.back() == "camera-a");
  assert(runtime.inject_eos("camera-a"));
  assert(failures.size() == 1);
  assert(failures.front().camera == "camera-a");
  assert(failures.front().category == "eos");
  assert(failures.front().scope == seeon::FailureScope::kSourceLocal);
  assert(runtime.remove("camera-a"));
  return 0;
}
