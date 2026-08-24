#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace seeon {
inline int unix_socket(int type) {
  const int descriptor = socket(AF_UNIX, type | SOCK_CLOEXEC, 0);
  if (descriptor < 0) {
    throw std::runtime_error{"AF_UNIX socket creation failed"};
  }
  return descriptor;
}

inline sockaddr_un socket_address(const std::filesystem::path& path) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string value = path.string();
  if (value.size() >= sizeof(address.sun_path)) {
    throw std::runtime_error{"AF_UNIX path is too long"};
  }
  std::memcpy(address.sun_path, value.c_str(), value.size() + 1);
  return address;
}
}  // namespace seeon
