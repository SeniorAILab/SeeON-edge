"""Resilient metadata wake consumer over inherited or owner-only AF_UNIX datagrams."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
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
        endpoint: object,
        slot: LatestMetadataSlot,
        puller: MetadataPuller,
    ) -> None:
        if not isinstance(endpoint, socket.socket):
            raise TypeError("inherited datagram wake socket required")
        self._endpoint = endpoint
        self._slot = slot
        self._puller = puller
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._fatal = threading.Event()
        self._binding_handler: Callable[[SourceBinding, MetadataFrame], None] | None = None

    @property
    def fatal_event(self) -> threading.Event:
        return self._fatal

    def __enter__(self) -> MetadataReceiver:
        self._socket = self._endpoint
        self._thread = threading.Thread(target=self._run, name="deepstream-metadata", daemon=True)
        self._thread.start()
        return self

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
            if data == b"":
                self._slot.mark_malformed()
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
    def close(self) -> None:
        self._stopping.set()
        receiver = self._socket
        if receiver is not None:
            receiver.shutdown(socket.SHUT_RD)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if receiver is not None:
            receiver.close()
        self._socket = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "MetadataPullFailure",
    "MetadataPuller",
    "MetadataPullStopped",
    "MetadataReceiver",
]
