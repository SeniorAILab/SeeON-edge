from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Self, final

from typing_extensions import override

from worker.runtime.state_dir import resolve_state_dir as _default_state_dir

GPU_LEASE_FILENAME: Final = ".gpu.lease"


@dataclass(slots=True)
class GpuLeaseUnavailableError(RuntimeError):
    lease_path: Path

    @override
    def __str__(self) -> str:
        return (
            f"GPU lease is already held by another process: {self.lease_path.name}; "
            "refusing to start before any CUDA/NVDEC/model construction"
        )


def resolve_state_dir(state_dir: Path | None = None) -> Path:
    if state_dir is not None:
        return state_dir
    return _default_state_dir()


@final
class GpuLease:
    def __init__(self, handle: BinaryIO, lease_path: Path) -> None:
        self._handle: BinaryIO | None = handle
        self._lease_path = lease_path

    @property
    def lease_path(self) -> Path:
        return self._lease_path

    @property
    def held(self) -> bool:
        return self._handle is not None

    @classmethod
    def acquire(cls, state_dir: Path | None = None) -> Self:
        directory = resolve_state_dir(state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        lease_path = directory / GPU_LEASE_FILENAME
        descriptor = os.open(
            lease_path,
            os.O_CLOEXEC | os.O_CREAT | os.O_RDWR,
            0o600,
        )
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise GpuLeaseUnavailableError(lease_path) from None
            raise
        _record_owner(handle)
        return cls(handle, lease_path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _record_owner(handle: BinaryIO) -> None:
    try:
        _ = handle.seek(0)
        handle.truncate(0)
        _ = handle.write(f"{os.getpid()}\n".encode())
        handle.flush()
    except OSError:
        return


__all__ = [
    "GPU_LEASE_FILENAME",
    "GpuLease",
    "GpuLeaseUnavailableError",
    "resolve_state_dir",
]
