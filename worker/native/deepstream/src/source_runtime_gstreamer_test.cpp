#include "source_runtime.hpp"

#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

struct CallbackControl {
  std::mutex mutex;
  std::condition_variable changed;
  bool release = false;
  std::size_t host_entries = 0;
  std::size_t host_exits = 0;
  std::size_t device_entries = 0;
  std::vector<seeon::NativeFailure> failures;
};

bool wait_for(CallbackControl& control,
              const std::function<bool(const CallbackControl&)>& condition) {
  std::unique_lock lock(control.mutex);
  return control.changed.wait_for(lock, std::chrono::seconds{4},
                                  [&] { return condition(control); });
}

seeon::SourceRuntime make_loopback_runtime(CallbackControl& control) {
  return seeon::SourceRuntime(
      [&control](const std::string&, const seeon::PipelineBindingPtr&,
                 const seeon::HostFrameView&) {
        std::unique_lock lock(control.mutex);
        ++control.host_entries;
        control.changed.notify_all();
        static_cast<void>(
            control.changed.wait_for(lock, std::chrono::seconds{5},
                                     [&] { return control.release; }));
        ++control.host_exits;
        control.changed.notify_all();
      },
      [&control](const std::string&, const seeon::PipelineBindingPtr&,
                 const seeon::DeviceFrameView&) {
        std::lock_guard lock(control.mutex);
        ++control.device_entries;
        control.changed.notify_all();
      },
      [&control](const seeon::NativeFailure& failure) {
        std::lock_guard lock(control.mutex);
        control.failures.push_back(failure);
        control.changed.notify_all();
      });
}
}  // namespace

int main() {
  gst_init(nullptr, nullptr);
  const seeon::NativeFailure eos =
      seeon::classify_bus_failure(true, 0, 0, "uridecodebin", "camera-a");
  check(eos.category == "eos" && eos.scope == seeon::FailureScope::kSourceLocal,
        "real-bus EOS classification changed");

  const seeon::NativeFailure camera_message = seeon::classify_bus_failure(
      false, GST_STREAM_ERROR, GST_STREAM_ERROR_FAILED, "uridecodebin", "cuda.context.camera");
  check(camera_message.scope == seeon::FailureScope::kSourceLocal,
        "camera text incorrectly changed fatal scope");

  const seeon::NativeFailure shared = seeon::classify_bus_failure(
      false, GST_LIBRARY_ERROR, GST_LIBRARY_ERROR_FAILED, "seeonperceptiontransform", "camera-a");
  check(shared.scope == seeon::FailureScope::kFatal && shared.category == "shared_pipeline",
        "shared GStreamer failure was not fatal");

  CallbackControl drain_control;
  auto drain_runtime = make_loopback_runtime(drain_control);
  std::string error_code;
  check(drain_runtime.add("drain-camera", "loopback://drain-camera",
                          std::make_shared<seeon::PipelineBinding>(1, 1), &error_code),
        "loopback source add failed");
  check(wait_for(drain_control,
                 [](const CallbackControl& control) { return control.host_entries == 1; }),
        "loopback host callback did not enter");

  bool quiesce_returned = false;
  bool quiesce_result = false;
  bool quiesce_started = false;
  std::thread quiesce_thread([&] {
    {
      std::lock_guard lock(drain_control.mutex);
      quiesce_started = true;
      drain_control.changed.notify_all();
    }
    const bool result = drain_runtime.quiesce("drain-camera");
    std::lock_guard lock(drain_control.mutex);
    quiesce_result = result;
    quiesce_returned = true;
    drain_control.changed.notify_all();
  });
  check(wait_for(drain_control, [&](const CallbackControl&) { return quiesce_started; }),
        "quiesce worker did not start");
  {
    std::lock_guard lock(drain_control.mutex);
    check(!quiesce_returned, "quiesce ACKed while host callback was in flight");
    drain_control.release = true;
    drain_control.changed.notify_all();
  }
  check(wait_for(drain_control, [&](const CallbackControl&) { return quiesce_returned; }),
        "quiesce did not complete after host callback release");
  quiesce_thread.join();
  check(quiesce_result, "quiesce failed after host callback drained");
  {
    std::lock_guard lock(drain_control.mutex);
    check(drain_control.failures.empty(), "successful drain emitted a fatal failure");
  }

  check(drain_runtime.restart("drain-camera", std::make_shared<seeon::PipelineBinding>(2, 2),
                              &error_code),
        "source restart failed after drain");
  check(wait_for(drain_control,
                 [](const CallbackControl& control) { return control.host_entries >= 2; }),
        "restarted source did not invoke the host callback");
  {
    std::lock_guard lock(drain_control.mutex);
    check(drain_control.device_entries == 0, "loopback source invoked device callback");
  }
  check(drain_runtime.remove("drain-camera"), "restarted source cleanup failed");

  constexpr std::size_t stress_cycles = 50;
  CallbackControl stress_control;
  {
    std::lock_guard lock(stress_control.mutex);
    stress_control.release = true;
  }
  auto stress_runtime = make_loopback_runtime(stress_control);
  for (std::size_t cycle = 0; cycle < stress_cycles; ++cycle) {
    const auto first_callback = cycle * 2 + 1;
    const auto second_callback = first_callback + 1;
    check(stress_runtime.add("stress-camera", "loopback://stress-camera",
                             std::make_shared<seeon::PipelineBinding>(cycle * 2 + 10,
                                                                       cycle * 2 + 10),
                             &error_code),
          "stress loopback source add failed");
    check(wait_for(stress_control, [&](const CallbackControl& control) {
            return control.host_entries >= first_callback;
          }),
          "stress loopback source did not invoke the initial host callback");
    check(stress_runtime.quiesce("stress-camera"), "stress source quiesce failed");
    check(stress_runtime.restart(
              "stress-camera",
              std::make_shared<seeon::PipelineBinding>(cycle * 2 + 11, cycle * 2 + 11),
              &error_code),
          "stress source restart failed");
    check(wait_for(stress_control, [&](const CallbackControl& control) {
            return control.host_entries >= second_callback;
          }),
          "stress restarted source did not invoke the host callback");
    check(stress_runtime.remove("stress-camera"), "stress source cleanup failed");
  }
  {
    std::lock_guard lock(stress_control.mutex);
    check(stress_control.host_entries >= stress_cycles * 2 &&
              stress_control.host_exits == stress_control.host_entries,
          "stress lifecycle did not drain every host callback");
    check(stress_control.device_entries == 0, "stress loopback source invoked device callback");
    check(stress_control.failures.empty(), "successful stress lifecycle emitted a failure");
  }

  CallbackControl timeout_control;
  auto timeout_runtime = make_loopback_runtime(timeout_control);
  check(timeout_runtime.add("timeout-camera", "loopback://timeout-camera",
                            std::make_shared<seeon::PipelineBinding>(3, 3), &error_code),
        "timeout loopback source add failed");
  check(wait_for(timeout_control,
                 [](const CallbackControl& control) { return control.host_entries == 1; }),
        "timeout loopback host callback did not enter");

  bool timeout_quiesce_returned = false;
  bool timeout_quiesce_result = true;
  bool timeout_quiesce_started = false;
  std::chrono::steady_clock::time_point timeout_quiesce_start;
  std::thread timeout_quiesce_thread([&] {
    {
      std::lock_guard lock(timeout_control.mutex);
      timeout_quiesce_start = std::chrono::steady_clock::now();
      timeout_quiesce_started = true;
      timeout_control.changed.notify_all();
    }
    const bool result = timeout_runtime.quiesce("timeout-camera");
    std::lock_guard lock(timeout_control.mutex);
    timeout_quiesce_result = result;
    timeout_quiesce_returned = true;
    timeout_control.changed.notify_all();
  });
  check(wait_for(timeout_control,
                 [&](const CallbackControl&) { return timeout_quiesce_started; }),
        "timeout quiesce worker did not start");
  check(wait_for(timeout_control,
                 [&](const CallbackControl&) { return timeout_quiesce_returned; }),
        "quiesce did not return at its drain deadline");
  timeout_quiesce_thread.join();
  const auto timeout_quiesce_elapsed = std::chrono::steady_clock::now() - timeout_quiesce_start;
  check(!timeout_quiesce_result, "quiesce succeeded while host callback exceeded deadline");
  check(timeout_quiesce_elapsed >= std::chrono::milliseconds{1800} &&
            timeout_quiesce_elapsed < std::chrono::seconds{3},
        "quiesce did not honor the two-second total drain deadline");
  {
    std::lock_guard lock(timeout_control.mutex);
    check(!timeout_control.release && timeout_control.host_exits == 0,
          "quiesce returned after the blocked host callback was released");
    check(timeout_control.failures.size() == 1 &&
              timeout_control.failures.front().camera == "timeout-camera" &&
              timeout_control.failures.front().category == "inference_drain_timeout" &&
              timeout_control.failures.front().scope == seeon::FailureScope::kFatal,
          "timeout drain did not emit exactly one fatal inference_drain_timeout");
  }
  {
    std::lock_guard lock(timeout_control.mutex);
    timeout_control.release = true;
    timeout_control.changed.notify_all();
  }
  check(timeout_runtime.remove("timeout-camera"), "timed out source did not clean up after release");
  {
    std::lock_guard lock(timeout_control.mutex);
    check(timeout_control.failures.size() == 1,
          "safe cleanup emitted a second inference_drain_timeout");
    check(timeout_control.device_entries == 0, "timeout loopback source invoked device callback");
  }
  return 0;
}
