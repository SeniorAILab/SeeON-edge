#include "au_transport.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) { std::cerr << message << '\n'; std::abort(); }
}
std::vector<std::uint8_t> receive_exact(int descriptor, std::size_t size) {
  std::vector<std::uint8_t> bytes(size);
  std::size_t offset = 0;
  while (offset < size) {
    const auto count = recv(descriptor, bytes.data() + offset, size - offset, 0);
    check(count > 0, "AU sender stream ended early");
    offset += static_cast<std::size_t>(count);
  }
  return bytes;
}
seeon::AuEnvelope envelope(std::uint64_t sequence, std::size_t size) {
  return {"camera-a", 1, 1, sequence,
          {seeon::AuCodec::kH264, seeon::AuFraming::kAnnexB, 1, 1, 1, 1, 90000,
           640, 360, sequence == 1, "caps", {}, std::vector<std::uint8_t>(size, 0x41)}};
}
constexpr std::size_t kAuHeaderBytes = 84;

// Packed header layout: magic[4], body_size u32, kind u8, codec u8,
// framing u8, keyframe u8, generation u32, epoch u64 at offset 16.
std::size_t body_size_of(const std::vector<std::uint8_t>& header) {
  return static_cast<std::size_t>(header[4]) |
         static_cast<std::size_t>(header[5]) << 8U |
         static_cast<std::size_t>(header[6]) << 16U |
         static_cast<std::size_t>(header[7]) << 24U;
}

std::uint64_t decode_epoch(const std::vector<std::uint8_t>& header) {
  std::uint64_t epoch = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    epoch |= static_cast<std::uint64_t>(header[16 + index]) << (8U * index);
  }
  return epoch;
}

seeon::AuEnvelope envelope_for(std::uint64_t epoch, std::uint64_t sequence,
                                std::size_t size) {
  return {"camera-a", 1, epoch, sequence,
          {seeon::AuCodec::kH264, seeon::AuFraming::kAnnexB, 1, 1, 1, 1, 90000,
           640, 360, sequence == 1, "caps", {}, std::vector<std::uint8_t>(size, 0x41)}};
}
}  // namespace

int main() {
  int descriptors[2]{};
  check(socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) == 0, "socketpair failed");
  int send_buffer = 4096;
  setsockopt(descriptors[0], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
  seeon::AuSender sender(descriptors[0], 2, seeon::kMaxAuFrameBytes);
  std::uint64_t sequence = 1;
  while (sequence < 100 && sender.enqueue(envelope(sequence, 1024 * 1024))) ++sequence;
  check(sequence < 100, "bounded sender never declared overflow");
  for (std::uint64_t expected = 1; expected <= sequence; ++expected) {
    const auto header = receive_exact(descriptors[1], 84);
    const auto body_size = static_cast<std::size_t>(header[4]) |
                           static_cast<std::size_t>(header[5]) << 8U |
                           static_cast<std::size_t>(header[6]) << 16U |
                           static_cast<std::size_t>(header[7]) << 24U;
    check(header[8] == (expected == sequence ? 2 : 1), "gap cut ahead of accepted AUs");
    check(receive_exact(descriptors[1], body_size).size() == body_size, "AU body truncated");
  }
  check(!sender.enqueue(envelope(999, seeon::kMaxAuFrameBytes + 1)),
        "oversized AU crossed the 32 MiB boundary");
  sender.stop();
  close(descriptors[0]);
  close(descriptors[1]);

  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    seeon::AuSender clean(pair[0], 4, seeon::kMaxAuFrameBytes);
    check(clean.enqueue(envelope_for(2, 1, 64)),
          "control: a 64-byte epoch-transition envelope must be admitted by a "
          "sender whose gap slot is free");
    clean.stop();
    close(pair[1]);
  }

  // A new stream epoch begins at sequence 1 because PipelineBinding is
  // recreated on a source rebuild. The receiver treats the first unit of an
  // epoch as the carrier of the transition, and rejects the epoch as
  // discontinuous if it never arrives.
  //
  // But the gap slot holds exactly one envelope, and while it is occupied
  // EVERY enqueue is refused - including that first unit. A rebuild burst
  // fills the queue, the overflow claims the single gap slot, and the very
  // next thing produced is epoch N+1 sequence 1, which is discarded. The
  // receiver then sees the epoch beginning at sequence 2 and reports a
  // discontinuity, which requests another rebuild, which fills the queue
  // again.
  //
  // Observed on the live 13-camera stack: rings frozen at active_epoch=1
  // while triggers reached 5, and clip selection failing with 'trigger
  // stream epoch is no longer active'. This test pins the mechanism without
  // asserting a remedy, because the remedy is a contract decision about how
  // an epoch transition survives backpressure.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int small = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    seeon::AuSender gapped(pair[0], 1, seeon::kMaxAuFrameBytes);

    // Fill the queue and then overflow it, so the gap slot is claimed.
    std::uint64_t filled = 0;
    for (std::uint64_t index = 1; index <= 64; ++index) {
      if (!gapped.enqueue(envelope_for(1, index, 8192))) {
        filled = index;
        break;
      }
    }
    check(filled != 0, "sender never overflowed, cannot exercise the gap slot");
    const auto dropped_after_overflow = gapped.dropped();
    check(dropped_after_overflow > 0, "overflow did not record a drop");

    // The source rebuilds: epoch 2 starts its sequence at 1. This envelope is
    // 64 bytes, far below every size limit, so its rejection can only be the
    // occupied gap slot -- a cumulative dropped() count alone would not
    // establish that, which is why the size is chosen to rule out the other
    // refusal conditions.
    // Still refused - backpressure is real and the unit is not queued.
    const bool transition_admitted = gapped.enqueue(envelope_for(2, 1, 64));
    check(!transition_admitted, "an overflowing sender must still refuse");
    check(gapped.dropped() == dropped_after_overflow + 1,
          "the epoch transition was refused without being counted as dropped");

    // But it must not be LOST: the gap slot now carries the newest epoch, so
    // the receiver learns about the transition instead of seeing the new
    // epoch begin at sequence 2 and stranding on the old one.
    //
    // Drained with a receive timeout rather than receive_exact, which aborts
    // on EOF; the sender is still running so the gap is flushed behind the
    // queued units.
    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    bool saw_epoch_two_gap = false;
    for (int index = 0; index < 256 && !saw_epoch_two_gap; ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) {
          complete = false;
          break;
        }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> sink_body(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], sink_body.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      if (header[8] == 2 && decode_epoch(header) == 2) saw_epoch_two_gap = true;
    }
    gapped.stop();
    check(saw_epoch_two_gap,
          "the epoch transition was destroyed by backpressure: the gap slot "
          "must carry the newest epoch so the receiver is not stranded (#429)");
    gapped.stop();
    close(pair[1]);
  }

  return 0;
}
