#include "au_transport.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <utility>

namespace seeon {
namespace {
constexpr std::array<char, 4> kMagic{'S', 'A', 'U', '1'};
constexpr std::uint8_t kAccessUnit = 1;
constexpr std::uint8_t kGap = 2;
#pragma pack(push, 1)
struct Header {
  std::array<char, 4> magic;
  std::uint32_t body_size;
  std::uint8_t kind;
  std::uint8_t codec;
  std::uint8_t framing;
  std::uint8_t keyframe;
  std::uint32_t generation;
  std::uint64_t epoch;
  std::uint64_t sequence;
  std::int64_t pts;
  std::int64_t dts;
  std::int64_t duration;
  std::int32_t time_base_num;
  std::int32_t time_base_den;
  std::uint32_t width;
  std::uint32_t height;
  std::uint16_t camera_size;
  std::uint16_t caps_size;
  std::uint32_t codec_data_size;
  std::uint32_t payload_size;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 84);

std::vector<std::uint8_t> encode(const AuEnvelope& envelope, bool gap) {
  const auto& unit = envelope.unit;
  Header header{kMagic,
                0,
                gap ? kGap : kAccessUnit,
                static_cast<std::uint8_t>(unit.codec),
                static_cast<std::uint8_t>(unit.framing),
                static_cast<std::uint8_t>(unit.keyframe),
                envelope.generation,
                envelope.epoch,
                envelope.sequence,
                unit.pts,
                unit.dts,
                unit.duration,
                unit.time_base_num,
                unit.time_base_den,
                unit.width,
                unit.height,
                static_cast<std::uint16_t>(envelope.camera.size()),
                static_cast<std::uint16_t>(unit.parser_caps.size()),
                gap ? 0U : static_cast<std::uint32_t>(unit.codec_data.size()),
                gap ? 0U : static_cast<std::uint32_t>(unit.payload.size())};
  header.body_size = header.camera_size + header.caps_size + header.codec_data_size +
                     header.payload_size;
  std::vector<std::uint8_t> bytes(sizeof(header) + header.body_size);
  std::memcpy(bytes.data(), &header, sizeof(header));
  auto* output = bytes.data() + sizeof(header);
  if (header.camera_size > 0) {
    std::memcpy(output, envelope.camera.data(), header.camera_size);
    output += header.camera_size;
  }
  if (header.caps_size > 0) {
    std::memcpy(output, unit.parser_caps.data(), header.caps_size);
    output += header.caps_size;
  }
  if (!gap && header.codec_data_size > 0) {
    std::memcpy(output, unit.codec_data.data(), header.codec_data_size);
    output += header.codec_data_size;
  }
  if (!gap && header.payload_size > 0) {
    std::memcpy(output, unit.payload.data(), header.payload_size);
  }
  return bytes;
}

bool send_all(int descriptor, const std::vector<std::uint8_t>& bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = send(descriptor, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}
}  // namespace

PipelineBinding::PipelineBinding(std::uint32_t generation, std::uint64_t epoch)
    : generation_(generation), epoch_(epoch) {}
void PipelineBinding::invalidate() { std::lock_guard lock{mutex_}; live_ = false; }
bool PipelineBinding::dispatch_au(
    const std::function<void(std::uint32_t, std::uint64_t, std::uint64_t)>& action) {
  std::lock_guard lock{mutex_};
  if (!live_) return false;
  action(generation_, epoch_, ++sequence_);
  return true;
}
bool PipelineBinding::dispatch_frame(
    const std::function<void(std::uint32_t, std::uint64_t)>& action) {
  std::lock_guard lock{mutex_};
  if (!live_) return false;
  action(generation_, epoch_);
  return true;
}

AuSender::AuSender(int descriptor, std::size_t max_items, std::size_t max_bytes)
    : descriptor_(descriptor), max_items_(max_items), max_bytes_(max_bytes),
      thread_(&AuSender::run, this) {}
AuSender::~AuSender() { stop(); }

bool AuSender::enqueue(AuEnvelope envelope) {
  const auto size = envelope.camera.size() + envelope.unit.parser_caps.size() +
                    envelope.unit.payload.size() + envelope.unit.codec_data.size();
  std::lock_guard lock{mutex_};
  if (stopped_) return false;
  // Validity limits are absolute: an envelope that cannot legally be framed is
  // refused outright and is never eligible for reservation, because a reserved
  // unit is later emitted as an ordinary access unit and would otherwise
  // bypass the very limit that rejected it.
  const bool invalid = size > kMaxAuFrameBytes || size > max_bytes_ ||
                       envelope.camera.size() > UINT16_MAX ||
                       envelope.unit.parser_caps.size() > UINT16_MAX;
  const bool congested =
      gap_.has_value() || queue_.size() >= max_items_ || bytes_ + size > max_bytes_;
  if (invalid || congested) {
    ++dropped_;
    // A unit that opens a stream epoch is reserved rather than destroyed, but
    // only when it was refused for congestion. The gap slot holds one envelope
    // and, while occupied, refuses everything - including the new epoch's
    // sequence 1, whose loss strands the receiver on the dead epoch
    // permanently (#429). Everything enqueued here is an access unit; gap
    // markers are synthesised at send time, so sequence 1 is exactly the unit
    // that opens an epoch.
    //
    // Held per camera and keyed by (generation, epoch) so a higher generation
    // wins even though its epoch restarts at 1 - this sender is shared by every
    // camera, so a scalar epoch comparison could not express that. Drained
    // ahead of the gap as an ORDINARY unit, which is what makes it effective:
    // the receiver adopts an epoch from a normal sequence-1 unit and discards
    // gap markers before adoption.
    if (!invalid && envelope.sequence == 1) {
      const auto existing = transitions_.find(envelope.camera);
      const bool newer =
          existing == transitions_.end() ||
          std::make_pair(envelope.generation, envelope.epoch) >
              std::make_pair(existing->second.generation, existing->second.epoch);
      if (newer) transitions_[envelope.camera] = std::move(envelope);
      ready_.notify_one();
      return false;
    }
    if (!gap_.has_value()) gap_ = std::move(envelope);
    ready_.notify_one();
    return false;
  }
  bytes_ += size;
  queue_.push_back(std::move(envelope));
  ready_.notify_one();
  return true;
}

void AuSender::run() {
  while (true) {
    AuEnvelope envelope{};
    bool gap = false;
    {
      std::unique_lock lock{mutex_};
      ready_.wait(lock, [this] {
        return stopped_ || gap_.has_value() || !queue_.empty() || !transitions_.empty();
      });
      if (stopped_ && !gap_.has_value() && queue_.empty() && transitions_.empty()) return;
      if (!queue_.empty()) {
        envelope = std::move(queue_.front());
        bytes_ -= envelope.camera.size() + envelope.unit.parser_caps.size() +
                  envelope.unit.payload.size() + envelope.unit.codec_data.size();
        queue_.pop_front();
      } else if (!transitions_.empty()) {
        // Ahead of the gap, and as an ordinary unit: the receiver adopts an
        // epoch from a normal sequence-1 access unit, never from a gap marker.
        const auto first = transitions_.begin();
        envelope = std::move(first->second);
        transitions_.erase(first);
      } else {
        envelope = std::move(*gap_);
        gap_.reset();
        gap = true;
      }
    }
    if (!send_all(descriptor_, encode(envelope, gap))) {
      std::lock_guard lock{mutex_};
      stopped_ = true;
      queue_.clear();
      gap_.reset();
      transitions_.clear();
      return;
    }
  }
}

void AuSender::stop() {
  bool shutdown_required = false;
  {
    std::lock_guard lock{mutex_};
    shutdown_required = !stopped_;
    stopped_ = true;
  }
  if (shutdown_required) shutdown(descriptor_, SHUT_WR);
  ready_.notify_all();
  if (thread_.joinable()) thread_.join();
}
std::uint64_t AuSender::dropped() const { std::lock_guard lock{mutex_}; return dropped_; }
}  // namespace seeon
