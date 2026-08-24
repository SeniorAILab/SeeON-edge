#include "child_server.hpp"

#include <charconv>
#include <iostream>
#include <optional>
#include <string_view>

namespace {
std::optional<std::array<std::uint8_t, 16>> parse_uuid(std::string_view value) {
  std::string compact;
  compact.reserve(32);
  for (const char character : value) {
    if (character != '-') {
      compact.push_back(character);
    }
  }
  if (compact.size() != 32) {
    return std::nullopt;
  }
  std::array<std::uint8_t, 16> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    unsigned int parsed = 0;
    const auto first = compact.data() + index * 2;
    const auto result = std::from_chars(first, first + 2, parsed, 16);
    if (result.ec != std::errc{} || result.ptr != first + 2) {
      return std::nullopt;
    }
    bytes[index] = static_cast<std::uint8_t>(parsed);
  }
  return bytes;
}

std::optional<seeon::ChildOptions> parse_options(int argc, char** argv) {
  if (argc != 15) {
    return std::nullopt;
  }
  seeon::ChildOptions options{};
  for (int index = 1; index < argc; index += 2) {
    const std::string_view flag{argv[index]};
    const std::string value{argv[index + 1]};
    if (flag == "--control-socket") {
      options.control_socket = value;
    } else if (flag == "--metadata-socket") {
      options.metadata_socket = value;
    } else if (flag == "--boot-id") {
      const auto boot = parse_uuid(value);
      if (!boot.has_value()) {
        return std::nullopt;
      }
      options.worker_boot_id = *boot;
    } else if (flag == "--gpu-id") {
      options.gpu_id = value;
    } else if (flag == "--child-id") {
      const auto child = parse_uuid(value);
      if (!child.has_value()) {
        return std::nullopt;
      }
      options.child_instance_id = *child;
    } else if (flag == "--first-fault") {
      options.first_fault = value;
    } else if (flag == "--ready-fd") {
      const auto result = std::from_chars(value.data(), value.data() + value.size(), options.ready_fd);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
  if (options.control_socket.empty() || options.metadata_socket.empty() || options.gpu_id.empty() ||
      options.first_fault.empty() || options.ready_fd < 0) {
    return std::nullopt;
  }
  return options;
}
}  // namespace

int main(int argc, char** argv) {
  const auto options = parse_options(argc, argv);
  if (!options.has_value()) {
    std::cerr << "usage: seeon-deepstream-child --control-socket PATH --metadata-socket PATH "
                 "--boot-id UUID --gpu-id ID --child-id UUID --first-fault PATH --ready-fd FD\n";
    return 2;
  }
  return seeon::ChildServer{*options}.run();
}
