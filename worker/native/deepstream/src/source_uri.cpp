#include "source_runtime.hpp"

#include <algorithm>

namespace seeon {
bool valid_source_uri(const std::string& uri) {
  if (!(uri.starts_with("rtsp://") || uri.starts_with("loopback://"))) return false;
  return std::ranges::none_of(uri, [](unsigned char character) {
    return character < 32 || character == 127;
  });
}
}  // namespace seeon
