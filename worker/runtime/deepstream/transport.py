"""Inherited socketpair and identity-pipe spawn for one native child."""

from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

from worker.runtime.deepstream.config import ChildConfig


@dataclass(frozen=True, slots=True)
class SpawnRequest:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    pass_fds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChildTransport:
    process: subprocess.Popen[bytes]
    control: socket.socket
    wake: socket.socket
    failures: socket.socket
    ready_fd: int


def spawn_process(request: SpawnRequest) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - image-owned executable
        request.command,
        env=request.environment,
        pass_fds=request.pass_fds,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def spawn_child(config: ChildConfig) -> ChildTransport:
    control_parent, control_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    wake_parent, wake_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    failure_parent, failure_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    identity_read, identity_write = os.pipe()
    ready_read, ready_write = os.pipe()
    identity = config.worker_boot_id.bytes + config.child_instance_id.bytes + os.getpid().to_bytes(
        4, "little"
    )
    _ = os.write(identity_write, identity)
    os.close(identity_write)
    command = (
        str(config.executable),
        "--control-fd",
        str(control_child.fileno()),
        "--wake-fd",
        str(wake_child.fileno()),
        "--failure-fd",
        str(failure_child.fileno()),
        "--identity-fd",
        str(identity_read),
        "--gpu-id",
        config.gpu_id,
        "--ready-fd",
        str(ready_write),
    )
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    try:
        process = spawn_process(
            SpawnRequest(
                command,
                child_env,
                (
                    control_child.fileno(),
                    wake_child.fileno(),
                    failure_child.fileno(),
                    identity_read,
                    ready_write,
                ),
            )
        )
    except OSError:
        control_parent.close()
        wake_parent.close()
        failure_parent.close()
        os.close(ready_read)
        raise
    finally:
        control_child.close()
        wake_child.close()
        failure_child.close()
        os.close(identity_read)
        os.close(ready_write)
    return ChildTransport(process, control_parent, wake_parent, failure_parent, ready_read)


__all__ = ["ChildTransport", "SpawnRequest", "spawn_child", "spawn_process"]
