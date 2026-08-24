#include "child_commands.hpp"

#include "unix_socket.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

namespace seeon {
namespace {
using ipc::Kind;
using ipc::Message;

#pragma pack(push, 1)
struct StatusPayload {
  std::uint64_t metadata_published;
  std::uint64_t metadata_overwritten;
  std::uint64_t wake_dropped;
  std::uint64_t source_failures;
  std::uint64_t malformed_frames;
  std::uint32_t source_count;
  std::uint8_t custom_transform_available;
};
#pragma pack(pop)

std::vector<std::uint8_t> text_payload(const std::string& value) {
  return {value.begin(), value.end()};
}

Message error_reply(const Message& request, const std::string& detail) {
  return ipc::reply(request, Kind::kError, text_payload(detail));
}

bool identity_matches(const ServerState& state, const Message& request) {
  return request.header.worker_boot_id == state.options.worker_boot_id &&
         request.header.child_instance_id == state.options.child_instance_id &&
         request.transform == "seeon-perception-v1";
}

void wake_metadata(ServerState& state, const std::string& camera) {
  const int sender = unix_socket(SOCK_DGRAM | SOCK_NONBLOCK);
  const sockaddr_un destination = socket_address(state.options.metadata_socket);
  const auto sent = sendto(sender, camera.data(), camera.size(), MSG_DONTWAIT,
                           reinterpret_cast<const sockaddr*>(&destination), sizeof(destination));
  if (sent < 0) {
    ++state.wake_dropped;
  }
  close(sender);
}

void publish_metadata(ServerState& state, const Message& request, SourceSlot& source) {
  ++source.source_sequence;
  ++state.publish_sequence;
  Message metadata = request;
  metadata.header.kind = static_cast<std::uint8_t>(Kind::kMetadata);
  metadata.header.stream_epoch = source.epoch;
  metadata.header.source_pts = request.header.source_pts == 0
                                   ? source.source_sequence * 1000
                                   : request.header.source_pts;
  metadata.header.source_sequence = source.source_sequence;
  metadata.header.native_publish_sequence = state.publish_sequence;
  metadata.header.request_id = 0;
  metadata.payload = {2, 2, 2, 0};
  const bool wake_required = !source.latest.has_value();
  if (!wake_required) {
    ++state.overwritten;
  }
  source.latest = std::move(metadata);
  ++state.published;
  if (wake_required) {
    wake_metadata(state, request.camera);
  }
  state.published_condition.notify_all();
}

Message status_reply(const ServerState& state, const Message& request) {
  const StatusPayload status{
      state.published,
      state.overwritten,
      state.wake_dropped,
      state.source_failures,
      state.malformed,
      static_cast<std::uint32_t>(state.runtime.count()),
      static_cast<std::uint8_t>(state.runtime.custom_transform_available()),
  };
  std::vector<std::uint8_t> payload(sizeof(status));
  std::memcpy(payload.data(), &status, sizeof(status));
  return ipc::reply(request, Kind::kStatusReply, std::move(payload));
}
}  // namespace

void ServerState::on_frame(const std::string& camera, std::uint64_t pts) {
  std::lock_guard lock{slot_mutex};
  const auto found = sources.find(camera);
  if (found == sources.end()) {
    return;
  }
  ipc::Message message{};
  message.header.worker_boot_id = options.worker_boot_id;
  message.header.child_instance_id = options.child_instance_id;
  message.header.source_generation = found->second.generation;
  message.header.stream_epoch = found->second.epoch;
  message.header.source_pts = pts;
  message.camera = camera;
  message.transform = "seeon-perception-v1";
  publish_metadata(*this, message, found->second);
}

CommandResult handle_command(ServerState& state, const ipc::Message& request) {
  if (!identity_matches(state, request)) {
    return {error_reply(request, "identity mismatch"), 4};
  }
  const auto kind = static_cast<ipc::Kind>(request.header.kind);
  std::string error;
  switch (kind) {
    case ipc::Kind::kAddSource: {
      {
        std::lock_guard lock{state.slot_mutex};
        const auto high_water = state.generation_high_water[request.camera];
        if (request.header.source_generation <= high_water ||
            state.sources.contains(request.camera)) {
          return {error_reply(request, "source generation rejected")};
        }
        state.generation_high_water[request.camera] = request.header.source_generation;
        state.sources.emplace(
            request.camera,
            SourceSlot{request.header.source_generation, 1, 0, std::nullopt});
      }
      const std::string uri{request.payload.begin(), request.payload.end()};
      if (!state.runtime.add(request.camera, uri, &error)) {
        std::lock_guard lock{state.slot_mutex};
        state.sources.erase(request.camera);
        return {error_reply(request, error)};
      }
      ipc::Message reply = ipc::reply(request, ipc::Kind::kEpochStarted);
      reply.header.stream_epoch = 1;
      return {std::move(reply)};
    }
    case ipc::Kind::kRemoveSource: {
      {
        std::lock_guard lock{state.slot_mutex};
        if (!state.sources.contains(request.camera)) {
          return {error_reply(request, "unknown source")};
        }
      }
      if (!state.runtime.remove(request.camera)) {
        return {error_reply(request, "source removal failed")};
      }
      std::lock_guard lock{state.slot_mutex};
      state.sources.erase(request.camera);
      return {ipc::reply(request, ipc::Kind::kAck)};
    }
    case ipc::Kind::kSourceFailure: {
      const std::string category{request.payload.begin(), request.payload.end()};
      const bool source_local = category == "rtsp_dns" || category == "rtsp_connect" ||
          category == "rtsp_auth" || category == "rtsp_timeout" || category == "rtsp_silence" ||
          category == "eos" || category == "depay" || category == "parser" || category == "caps" ||
          category == "decoder_source";
      if (!source_local) {
        return {error_reply(request, "ambiguous source failure"), 4};
      }
      std::uint64_t epoch = 0;
      {
        std::lock_guard lock{state.slot_mutex};
        const auto found = state.sources.find(request.camera);
        if (found == state.sources.end()) {
          return {error_reply(request, "unknown source")};
        }
        epoch = ++found->second.epoch;
        found->second.latest.reset();
        ++state.source_failures;
      }
      if (!state.runtime.rebuild(request.camera, &error)) {
        return {error_reply(request, error)};
      }
      ipc::Message reply = ipc::reply(request, ipc::Kind::kEpochStarted);
      reply.header.stream_epoch = epoch;
      return {std::move(reply)};
    }
    case ipc::Kind::kEmitMetadata: {
      std::lock_guard lock{state.slot_mutex};
      const auto found = state.sources.find(request.camera);
      if (found == state.sources.end()) {
        return {error_reply(request, "unknown source")};
      }
      publish_metadata(state, request, found->second);
      return {ipc::reply(request, ipc::Kind::kAck)};
    }
    case ipc::Kind::kGetLatest: {
      std::lock_guard lock{state.slot_mutex};
      const auto found = state.sources.find(request.camera);
      if (found == state.sources.end() || !found->second.latest.has_value()) {
        return {ipc::reply(request, ipc::Kind::kCapabilityInactive)};
      }
      ipc::Message latest = std::move(*found->second.latest);
      found->second.latest.reset();
      latest.header.request_id = request.header.request_id;
      return {std::move(latest)};
    }
    case ipc::Kind::kRecord:
    case ipc::Kind::kSnapshot:
      return {ipc::reply(request, ipc::Kind::kCapabilityInactive,
                         text_payload("dark capability not active"))};
    case ipc::Kind::kStatus: {
      std::lock_guard lock{state.slot_mutex};
      return {status_reply(state, request)};
    }
    case ipc::Kind::kFatal: {
      const std::string category{request.payload.begin(), request.payload.end()};
      const bool fatal = category == "cuda" || category == "xid" || category == "context" ||
                         category == "native_heap" || category == "tensorrt";
      return fatal ? CommandResult{ipc::reply(request, ipc::Kind::kAck), 4}
                   : CommandResult{error_reply(request, "unknown fatal category"), 4};
    }
    case ipc::Kind::kShutdown:
      return {ipc::reply(request, ipc::Kind::kAck), 0};
    case ipc::Kind::kWaitPublish: {
      if (request.payload.size() != sizeof(std::uint64_t)) {
        return {error_reply(request, "publish target size")};
      }
      std::uint64_t target = 0;
      std::memcpy(&target, request.payload.data(), sizeof(target));
      std::unique_lock lock{state.slot_mutex};
      const bool reached = state.published_condition.wait_for(
          lock,
          std::chrono::seconds{2},
          [&state, target] { return state.published >= target; });
      return reached ? CommandResult{ipc::reply(request, ipc::Kind::kAck)}
                     : CommandResult{error_reply(request, "publish target timeout"), 4};
    }
    default:
      return {error_reply(request, "unsupported command"), 4};
  }
}
}  // namespace seeon
