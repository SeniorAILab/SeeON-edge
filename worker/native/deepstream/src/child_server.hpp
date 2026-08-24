#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace seeon {
struct ChildOptions {
  std::filesystem::path control_socket;
  std::filesystem::path metadata_socket;
  std::array<std::uint8_t, 16> worker_boot_id;
  std::array<std::uint8_t, 16> child_instance_id;
  std::string gpu_id;
  std::filesystem::path first_fault;
  int ready_fd;
};

class ChildServer {
 public:
  explicit ChildServer(ChildOptions options);
  [[nodiscard]] int run();

 private:
  ChildOptions options_;
};
}  // namespace seeon
