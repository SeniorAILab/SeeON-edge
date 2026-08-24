#include "child_commands.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

seeon::ipc::Message request(const seeon::ChildOptions& options, seeon::ipc::Kind kind,
                            std::uint32_t generation, std::string payload = {}) {
  seeon::ipc::Message message{};
  message.header.kind = static_cast<std::uint8_t>(kind);
  message.header.worker_boot_id = options.worker_boot_id;
  message.header.child_instance_id = options.child_instance_id;
  message.header.source_generation = generation;
  message.camera = "camera-a";
  message.transform = "seeon-perception-v1";
  message.payload = {payload.begin(), payload.end()};
  return message;
}
}  // namespace

int main() {
  seeon::ChildOptions options{};
  options.worker_boot_id[0] = 1;
  options.child_instance_id[0] = 2;
  options.qa_mode = false;
  seeon::ServerState state{options};

  auto add = request(options, seeon::ipc::Kind::kAddSource, 1, "loopback://camera-a");
  auto result = seeon::handle_command(state, add);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kEpochStarted),
        "initial source add failed");
  check(result.reply.header.stream_epoch == 1, "initial epoch mismatch");

  auto emit = request(options, seeon::ipc::Kind::kEmitMetadata, 1);
  result = seeon::handle_command(state, emit);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kError),
        "synthetic metadata was accepted outside QA mode");

  auto remove = request(options, seeon::ipc::Kind::kRemoveSource, 1);
  result = seeon::handle_command(state, remove);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kAck),
        "source remove failed");

  add = request(options, seeon::ipc::Kind::kAddSource, 2, "loopback://camera-a");
  result = seeon::handle_command(state, add);
  check(result.reply.header.stream_epoch == 1, "re-add did not start a fresh epoch");

  auto rebuild = request(options, seeon::ipc::Kind::kSourceFailure, 2, "eos");
  result = seeon::handle_command(state, rebuild);
  check(result.reply.header.stream_epoch == 2, "reconnect did not advance epoch");

  result = seeon::handle_command(state, remove);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kAck),
        "final source remove failed");
  return 0;
}
