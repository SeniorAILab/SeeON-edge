"""Resilient metadata wake consumer over inherited or owner-only AF_UNIX datagrams."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, final, override

from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata_slot import LatestMetadataSlot, SourceBinding

_MAX_DATAGRAM: Final = 512


@dataclass(frozen=True, slots=True)
class MetadataPullFailure(Exception):
    code: str

    @override
    def __str__(self) -> str:
        return self.code


class MetadataPullStopped(Exception):
    """Confirmed child/control EOF; the dark runner must fail fast."""


class MetadataPuller(Protocol):
    def pull_latest(self, camera_id: str) -> MetadataFrame | None: ...
    def source_binding(self, camera_id: str) -> SourceBinding: ...


@final
class MetadataReceiver:
    """Pull latest metadata after wakeups; recover isolated malformed/pull failures."""

    def __init__(
        self,
        endpoint: Path | socket.socket,
        slot: LatestMetadataSlot,
        puller: MetadataPuller,
    ) -> None:
        self._endpoint = endpoint
        self._slot = slot
        self._puller = puller
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._received_count = 0
        self._enabled = threading.Event()
        self._enabled.set()
        self._stopping = threading.Event()
        self._fatal = threading.Event()
        self._binding_handler: Callable[[SourceBinding, MetadataFrame], None] | None = None

    @property
    def fatal_event(self) -> threading.Event:
        return self._fatal

    def __enter__(self) -> MetadataReceiver:
        match self._endpoint:
            case Path() as path:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(path.parent, 0o700)
                path.unlink(missing_ok=True)
                receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                receiver.bind(str(path))
                os.chmod(path, 0o600)
            case socket.socket() as inherited:
                receiver = inherited
        self._socket = receiver
        self._thread = threading.Thread(target=self._run, name="deepstream-metadata", daemon=True)
        self._thread.start()
        return self

    def _record_cycle(self) -> None:
        with self._condition:
            self._received_count += 1
            self._condition.notify_all()

    def set_binding_handler(
        self,
        handler: Callable[[SourceBinding, MetadataFrame], None],
    ) -> None:
        self._binding_handler = handler

    def pull_now(self, camera_id: str) -> MetadataFrame | None:
        if camera_id == "":
            self._slot.mark_malformed()
            return None
        try:
            metadata = self._puller.pull_latest(camera_id)
        except MetadataPullFailure:
            self._slot.mark_pull_failure()
            return None
        except MetadataPullStopped:
            self._fatal.set()
            raise
        if metadata is None:
            return None
        expected = self._slot.expected_binding(camera_id)
        advanced = expected is not None and metadata.identity.stream_epoch > expected.stream_epoch
        binding: SourceBinding | None = None
        if advanced:
            try:
                binding = self._puller.source_binding(camera_id)
            except MetadataPullFailure:
                self._slot.mark_pull_failure()
                return None
            self._slot.register_source(binding)
        accepted = self._slot.publish(metadata)
        if binding is not None and accepted and self._binding_handler is not None:
            self._binding_handler(binding, metadata)
        return metadata

    def _run(self) -> None:
        receiver = self._socket
        if receiver is None:
            return
        while not self._stopping.is_set():
            try:
                data = receiver.recv(_MAX_DATAGRAM)
            except OSError:
                if not self._stopping.is_set():
                    self._fatal.set()
                return
            if self._stopping.is_set():
                return
            _ = self._enabled.wait()
            if data == b"":
                self._slot.mark_malformed()
                self._record_cycle()
                continue
            try:
                camera_id = data.decode()
            except UnicodeDecodeError:
                self._slot.mark_malformed()
            else:
                try:
                    _ = self.pull_now(camera_id)
                except MetadataPullStopped:
                    return
            self._record_cycle()

    def pause(self) -> None:
        self._enabled.clear()

    def resume(self) -> None:
        self._enabled.set()

    def subscription(self) -> int:
        with self._condition:
            return self._received_count

    def wait_received(self, after: int, *, timeout_sec: float) -> None:
        with self._condition:
            received = self._condition.wait_for(
                lambda: self._received_count > after,
                timeout=timeout_sec,
            )
        if not received:
            raise TimeoutError("metadata receive deadline elapsed")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self._stopping.set()
        self._enabled.set()
        receiver = self._socket
        match self._endpoint:
            case Path() as path:
                sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                try:
                    _ = sender.sendto(b"\x00STOP", str(path))
                finally:
                    sender.close()
            case socket.socket():
                if receiver is not None:
                    receiver.shutdown(socket.SHUT_RD)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if receiver is not None:
            receiver.close()
        match self._endpoint:
            case Path() as path:
                path.unlink(missing_ok=True)
            case socket.socket():
                pass
        self._socket = None


__all__ = [
    "MetadataPullFailure",
    "MetadataPuller",
    "MetadataPullStopped",
    "MetadataReceiver",
]
