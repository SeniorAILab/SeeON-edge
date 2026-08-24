#pragma once

#include "child_server.hpp"
#include "ipc_protocol.hpp"
#include "source_runtime.hpp"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace seeon {
struct SourceSlot {
  std::uint32_t generation;
  std::uint64_t epoch;
  std::uint64_t source_sequence = 0;
  std::optional<ipc::Message> latest;
};

class ServerState {
 public:
  explicit ServerState(const ChildOptions& options_value)
      : options(options_value),
        runtime([this](const std::string& camera, std::uint64_t pts) { on_frame(camera, pts); }) {}

  void on_frame(const std::string& camera, std::uint64_t pts);

  const ChildOptions& options;
  SourceRuntime runtime;
  std::mutex slot_mutex;
  std::condition_variable published_condition;
  std::map<std::string, SourceSlot> sources;
  std::map<std::string, std::uint32_t> generation_high_water;
  std::uint64_t publish_sequence = 0;
  std::uint64_t published = 0;
  std::uint64_t overwritten = 0;
  std::uint64_t wake_dropped = 0;
  std::uint64_t source_failures = 0;
  std::uint64_t malformed = 0;
};

struct CommandResult {
  ipc::Message reply;
  int exit_code = -1;
};

[[nodiscard]] CommandResult handle_command(ServerState& state, const ipc::Message& request);
}  // namespace seeon
