"""Owner-only validation for dark-runner filesystem paths and stale QA sockets."""

from __future__ import annotations

import errno
import os
import socket
import stat
from pathlib import Path


class PrivatePathError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)
        self.code: str = code
        self.detail: str = detail


def remove_stale_socket(path: Path) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISSOCK(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        raise PrivatePathError("stale_socket_unsafe", path.name)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        probe.connect(str(path))
    except OSError as error:
        if error.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
            raise PrivatePathError("stale_socket_probe", path.name) from error
    else:
        raise PrivatePathError("stale_socket_live", path.name)
    finally:
        probe.close()
    path.unlink(missing_ok=True)


def validate_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid():
        raise PrivatePathError("runtime_directory_owner", path.name)
    if stat.S_IMODE(details.st_mode) != 0o700:
        raise PrivatePathError("runtime_directory_mode", oct(stat.S_IMODE(details.st_mode)))


__all__ = ["PrivatePathError", "remove_stale_socket", "validate_private_directory"]
