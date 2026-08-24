#include "child_server.hpp"

#include "child_commands.hpp"
#include "ipc_protocol.hpp"
#include "unix_socket.hpp"

#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <utility>
#include <vector>

namespace seeon {
ChildServer::ChildServer(ChildOptions options) : options_(std::move(options)) {}

int ChildServer::run() {
  if (prctl(PR_SET_PDEATHSIG, SIGTERM) < 0) {
    return 4;
  }
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGTERM);
  sigaddset(&signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  const int signal_fd = signalfd(-1, &signals, SFD_CLOEXEC);
  const int listener = unix_socket(SOCK_SEQPACKET);
  std::filesystem::create_directories(options_.control_socket.parent_path());
  std::filesystem::remove(options_.control_socket);
  const sockaddr_un address = socket_address(options_.control_socket);
  if (bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0 ||
      chmod(options_.control_socket.c_str(), 0600) < 0 || listen(listener, 1) < 0) {
    close(listener);
    close(signal_fd);
    return 4;
  }
  ServerState state{options_};
  if (!state.runtime.custom_transform_available()) {
    close(listener);
    close(signal_fd);
    std::filesystem::remove(options_.control_socket);
    return 4;
  }
  const auto ready_written = write(options_.ready_fd, "R", 1);
  close(options_.ready_fd);
  if (ready_written != 1) {
    close(listener);
    close(signal_fd);
    std::filesystem::remove(options_.control_socket);
    return 4;
  }
  pollfd startup[] = {{listener, POLLIN, 0}, {signal_fd, POLLIN, 0}};
  if (poll(startup, 2, -1) < 0 || startup[1].revents != 0) {
    close(listener);
    close(signal_fd);
    std::filesystem::remove(options_.control_socket);
    return 4;
  }
  const int client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
  close(listener);
  int exit_code = 4;
  std::vector<std::uint8_t> buffer(65'535);
  while (client >= 0) {
    pollfd active[] = {{client, POLLIN, 0}, {signal_fd, POLLIN, 0}};
    if (poll(active, 2, -1) < 0 || active[1].revents != 0) {
      break;
    }
    const auto received = recv(client, buffer.data(), buffer.size(), 0);
    if (received <= 0) {
      break;
    }
    std::vector<std::uint8_t> frame(buffer.begin(), buffer.begin() + received);
    const auto request = ipc::decode(frame);
    if (!request.has_value()) {
      ++state.malformed;
      break;
    }
    CommandResult result = handle_command(state, *request);
    const auto reply = ipc::encode(result.reply);
    static_cast<void>(send(client, reply.data(), reply.size(), MSG_NOSIGNAL));
    if (result.exit_code >= 0) {
      exit_code = result.exit_code;
      break;
    }
  }
  if (client >= 0) {
    close(client);
  }
  close(signal_fd);
  std::filesystem::remove(options_.control_socket);
  return exit_code;
}
}  // namespace seeon
