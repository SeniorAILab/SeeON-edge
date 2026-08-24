#include "ipc_protocol.hpp"
#include "perception_wire.hpp"

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <vector>

int main() {
  seeon::ipc::Message message{};
  message.header.kind = static_cast<std::uint8_t>(seeon::ipc::Kind::kAddSource);
  message.header.source_generation = 3;
  message.header.stream_epoch = 7;
  message.header.source_pts = 11;
  message.header.source_sequence = 12;
  message.header.native_publish_sequence = 13;
  message.header.request_id = 14;
  message.camera = "camera-a";
  message.transform = "seeon-perception-v1";
  message.payload = {1, 2, 3};

  const std::vector<std::uint8_t> encoded = seeon::ipc::encode(message);
  const auto decoded = seeon::ipc::decode(encoded);

  assert(decoded.has_value());
  assert(decoded->header.source_generation == 3);
  assert(decoded->header.stream_epoch == 7);
  assert(decoded->header.source_pts == 11);
  assert(decoded->header.source_sequence == 12);
  assert(decoded->header.native_publish_sequence == 13);
  assert(decoded->header.request_id == 14);
  assert(decoded->camera == "camera-a");
  assert(decoded->transform == "seeon-perception-v1");
  assert(decoded->payload == std::vector<std::uint8_t>({1, 2, 3}));

  std::vector<std::uint8_t> malformed = encoded;
  malformed[0] = 0;
  assert(!seeon::ipc::decode(malformed).has_value());

  message.header.worker_boot_id = {0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
                                   0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78};
  message.header.stream_epoch = 7;
  message.header.source_pts = 123456;
  message.header.source_sequence = 11;
  message.camera = "camera-a";
  std::ostringstream golden;
  for (const auto value : seeon::encode_empty_perception(message)) {
    golden << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
  }
  assert(golden.str() ==
         "5046563112345678123456781234567812345678080063616d6572612d61"
         "070000000000000040e20100000000000b0000000000000002020200000000000000");
  return 0;
}
