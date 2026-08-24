#include "perception_wire.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace seeon {
namespace {
template <typename Value>
void append(std::vector<std::uint8_t>& output, Value value) {
  const auto offset = output.size();
  output.resize(offset + sizeof(value));
  std::memcpy(output.data() + offset, &value, sizeof(value));
}
}  // namespace

std::vector<std::uint8_t> encode_empty_perception(const ipc::Message& envelope) {
  if (envelope.camera.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error{"camera identity too large"};
  }
  std::vector<std::uint8_t> payload{'P', 'F', 'V', '1'};
  payload.insert(payload.end(), envelope.header.worker_boot_id.begin(),
                 envelope.header.worker_boot_id.end());
  append(payload, static_cast<std::uint16_t>(envelope.camera.size()));
  payload.insert(payload.end(), envelope.camera.begin(), envelope.camera.end());
  append(payload, envelope.header.stream_epoch);
  append(payload, envelope.header.source_pts);
  append(payload, envelope.header.source_sequence);
  payload.insert(payload.end(), {2, 2, 2, 0});
  append(payload, static_cast<std::uint16_t>(0));
  append(payload, static_cast<std::uint16_t>(0));
  append(payload, static_cast<std::uint16_t>(0));
  return payload;
}
}  // namespace seeon
