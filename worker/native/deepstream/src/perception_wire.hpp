#pragma once

#include "ipc_protocol.hpp"

#include <cstdint>
#include <vector>

namespace seeon {
[[nodiscard]] std::vector<std::uint8_t> encode_empty_perception(const ipc::Message& envelope);
}  // namespace seeon
