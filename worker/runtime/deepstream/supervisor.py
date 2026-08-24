"""Python PID-1 ownership of dark native DeepStream child processes."""

from __future__ import annotations

import errno
import os
import select
import signal
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final, override

from worker.native.deepstream.control import (
    ChildControlError,
    ControlIdentity,
    DeepStreamControlClient,
)
from worker.native.deepstream.metadata import LatestMetadataSlot, MetadataReceiver
from worker.runtime.deepstream.config import (
    ChildConfig,
    DarkChildConfigError,
    configured_dark_supervisors,
)
from worker.runtime.deepstream.source_control import DarkSourceController
from worker.runtime.lease import GpuLease

_TRANSFORM_ID: Final = "seeon-perception-v1"


@dataclass(frozen=True, slots=True)
class ChildStartupError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ChildFatalError(Exception):
    gpu_id: str
    exit_code: int
    first_fault_path: Path

    @override
    def __str__(self) -> str:
        return f"DeepStream child gpu={self.gpu_id} exited {self.exit_code}"


def remove_stale_socket(path: Path) -> None:
    """Remove only a dead socket or non-socket residue; preserve a live owner."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        path.unlink()
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    probe.settimeout(0.1)
    try:
        probe.connect(str(path))
    except OSError as error:
        if error.errno not in (errno.ECONNREFUSED, errno.ENOENT):
            raise ChildStartupError("socket_probe", str(error)) from error
    else:
        raise ChildStartupError("socket_active", str(path))
    finally:
        probe.close()
    path.unlink(missing_ok=True)


@final
class DeepStreamChildSupervisor:
    """Own exactly one native child and its sockets for one GPU."""

    def __init__(self, config: ChildConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._metadata = LatestMetadataSlot()
        self._receiver: MetadataReceiver | None = None
        self._control: DeepStreamControlClient | None = None
        self._sources: DarkSourceController | None = None
        self._lease: GpuLease | None = None

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None or process.poll() is not None else process.pid

    @property
    def metadata(self) -> LatestMetadataSlot:
        return self._metadata

    @property
    def control(self) -> DeepStreamControlClient:
        control = self._control
        if control is None:
            raise ChildStartupError("not_started", self._config.gpu_id)
        return control

    @property
    def sources(self) -> DarkSourceController:
        sources = self._sources
        if sources is None:
            raise ChildStartupError("not_started", self._config.gpu_id)
        return sources

    @property
    def control_path(self) -> Path:
        return self._config.socket_dir / "control.sock"

    @property
    def metadata_path(self) -> Path:
        return self._config.socket_dir / "metadata.sock"

    def start(self) -> None:
        if self.pid is not None:
            raise ChildStartupError("already_started", self._config.gpu_id)
        self._config.socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._config.socket_dir, 0o700)
        remove_stale_socket(self.control_path)
        self._lease = GpuLease.acquire(self._config.lease_state_dir)
        ready_read, ready_write = os.pipe()
        command = (
            str(self._config.executable),
            "--control-socket",
            str(self.control_path),
            "--metadata-socket",
            str(self.metadata_path),
            "--boot-id",
            str(self._config.worker_boot_id),
            "--gpu-id",
            self._config.gpu_id,
            "--child-id",
            str(self._config.child_instance_id),
            "--first-fault",
            str(self._config.first_fault_path),
            "--ready-fd",
            str(ready_write),
        )
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = self._config.gpu_id
        try:
            self._process = subprocess.Popen(  # noqa: S603 - executable is image-owned config
                command,
                env=child_env,
                pass_fds=(ready_write,),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            os.close(ready_read)
            os.close(ready_write)
            self.stop()
            raise ChildStartupError("spawn_failed", str(error)) from error
        os.close(ready_write)
        try:
            readable, _, _ = select.select([ready_read], [], [], self._config.startup_timeout_sec)
            ready = os.read(ready_read, 1) if readable else b""
        finally:
            os.close(ready_read)
        if ready != b"R":
            self.stop()
            raise ChildStartupError("ready_timeout", self._config.gpu_id)
        control = DeepStreamControlClient(
            self.control_path,
            ControlIdentity(
                worker_boot_id=self._config.worker_boot_id,
                child_instance_id=self._config.child_instance_id,
                transform_id=_TRANSFORM_ID,
            ),
        )
        try:
            control.connect()
            _ = control.status()
            receiver = MetadataReceiver(self.metadata_path, self._metadata, control)
            _ = receiver.__enter__()
        except (ChildControlError, OSError) as error:
            control.close()
            self.stop()
            raise ChildStartupError("ready_probe_failed", str(error)) from error
        self._control = control
        self._receiver = receiver
        self._sources = DarkSourceController(control, self._metadata, receiver)

    def wait(self) -> int:
        process = self._process
        if process is None:
            raise ChildStartupError("not_started", self._config.gpu_id)
        code = process.wait()
        self._process = None
        if code != 0:
            self._persist_first_fault(code)
            raise ChildFatalError(self._config.gpu_id, code, self._config.first_fault_path)
        return code

    def _persist_first_fault(self, exit_code: int) -> None:
        path = self._config.first_fault_path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = (
            b"SDSF\x01"
            + self._config.worker_boot_id.bytes
            + self._config.child_instance_id.bytes
            + exit_code.to_bytes(4, "little", signed=True)
        )
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        try:
            _ = os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def stop(self) -> None:
        self._sources = None
        receiver, self._receiver = self._receiver, None
        if receiver is not None:
            receiver.__exit__(None, None, None)
        process = self._process
        control, self._control = self._control, None
        if process is not None and process.poll() is None:
            if control is not None:
                try:
                    control.shutdown()
                except ChildControlError:
                    process.send_signal(signal.SIGTERM)
            else:
                process.send_signal(signal.SIGTERM)
            try:
                _ = process.wait(timeout=self._config.stop_timeout_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = process.wait(timeout=self._config.stop_timeout_sec)
        if control is not None:
            control.close()
        self._process = None
        self.control_path.unlink(missing_ok=True)
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()


__all__ = [
    "ChildConfig",
    "ChildFatalError",
    "DarkChildConfigError",
    "ChildStartupError",
    "DeepStreamChildSupervisor",
    "configured_dark_supervisors",
    "remove_stale_socket",
]
