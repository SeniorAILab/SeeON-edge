#include "child_server.hpp"

#include <unistd.h>

#include <charconv>
#include <cstring>
#include <iostream>
#include <optional>
#include <string_view>

namespace {
bool parse_fd(std::string_view value, int* output) {
  const auto result = std::from_chars(value.data(), value.data() + value.size(), *output);
  return result.ec == std::errc{} && result.ptr == value.data() + value.size() && *output >= 0;
}

bool read_identity(int descriptor, seeon::ChildOptions* options) {
  std::array<std::uint8_t, 36> identity{};
  std::size_t offset = 0;
  while (offset < identity.size()) {
    const auto count = read(descriptor, identity.data() + offset, identity.size() - offset);
    if (count <= 0) {
      close(descriptor);
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  close(descriptor);
  std::memcpy(options->worker_boot_id.data(), identity.data(), 16);
  std::memcpy(options->child_instance_id.data(), identity.data() + 16, 16);
  std::memcpy(&options->parent_pid, identity.data() + 32, 4);
  return options->parent_pid > 0;
}

std::optional<seeon::ChildOptions> parse_options(int argc, char** argv) {
  if (argc != 13) {
    return std::nullopt;
  }
  seeon::ChildOptions options{};
  int identity_fd = -1;
  for (int index = 1; index < argc; index += 2) {
    const std::string_view flag{argv[index]};
    const std::string_view value{argv[index + 1]};
    if (flag == "--control-fd") {
      if (!parse_fd(value, &options.control_fd)) return std::nullopt;
    } else if (flag == "--wake-fd") {
      if (!parse_fd(value, &options.wake_fd)) return std::nullopt;
    } else if (flag == "--failure-fd") {
      if (!parse_fd(value, &options.failure_fd)) return std::nullopt;
    } else if (flag == "--identity-fd") {
      if (!parse_fd(value, &identity_fd)) return std::nullopt;
    } else if (flag == "--gpu-id") {
      options.gpu_id = value;
    } else if (flag == "--ready-fd") {
      if (!parse_fd(value, &options.ready_fd)) return std::nullopt;
    } else {
      return std::nullopt;
    }
  }
  if (options.control_fd < 0 || options.wake_fd < 0 || options.failure_fd < 0 ||
      options.ready_fd < 0 || identity_fd < 0 || options.gpu_id != "0" ||
      !read_identity(identity_fd, &options)) {
    return std::nullopt;
  }
  return options;
}
}  // namespace

int main(int argc, char** argv) {
  const auto options = parse_options(argc, argv);
  if (!options.has_value()) {
    std::cerr << "invalid inherited DeepStream child launch\n";
    return 2;
  }
  return seeon::ChildServer{*options}.run();
}
