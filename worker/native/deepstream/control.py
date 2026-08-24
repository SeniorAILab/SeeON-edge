"""Reliable long-lived AF_UNIX control session for the native child."""

from __future__ import annotations

import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final, override

from worker.native.deepstream.ipc import (
    ControlMessage,
    IpcProtocolError,
    MessageKind,
    MetadataFrame,
    decode_control_message,
    decode_metadata,
    encode_message,
)
from worker.native.deepstream.metadata import MetadataPullStopped, SourceBinding

_MAX_REPLY: Final = 65_535
_STATUS: Final = struct.Struct("<QQQQQIB")
_ACCEPTED_REPLIES: Final = frozenset(
    {
        MessageKind.ACK,
        MessageKind.STATUS_REPLY,
        MessageKind.EPOCH_STARTED,
        MessageKind.CAPABILITY_INACTIVE,
        MessageKind.METADATA,
    }
)
_DARK_CAPABILITIES: Final = frozenset({MessageKind.RECORD, MessageKind.SNAPSHOT})


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    worker_boot_id: uuid.UUID
    child_instance_id: uuid.UUID
    transform_id: str


@dataclass(frozen=True, slots=True)
class NativeStatus:
    metadata_published: int
    metadata_overwritten: int
    wake_dropped: int
    source_failures: int
    malformed_frames: int
    source_count: int
    custom_transform_available: bool


@dataclass(frozen=True, slots=True)
class ChildControlError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@final
class DeepStreamControlClient:
    """One correlated SOCK_SEQPACKET session; mutation tracks source generations."""

    def __init__(self, path: Path, identity: ControlIdentity, *, timeout_sec: float = 2.0) -> None:
        self._path = path
        self._identity = identity
        self._timeout_sec = timeout_sec
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._request_sequence = 0
        self._generations: dict[str, int] = {}
        self._epochs: dict[str, int] = {}

    def connect(self) -> None:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(self._timeout_sec)
        try:
            client.connect(str(self._path))
        except OSError as error:
            client.close()
            raise ChildControlError("control_connect", str(error)) from error
        self._socket = client

    def close(self) -> None:
        client, self._socket = self._socket, None
        if client is not None:
            client.close()

    def _message(self, kind: MessageKind, camera_id: str, payload: bytes = b"") -> ControlMessage:
        self._request_sequence += 1
        return ControlMessage(
            kind=kind,
            worker_boot_id=self._identity.worker_boot_id,
            child_instance_id=self._identity.child_instance_id,
            camera_id=camera_id,
            source_generation=self._generations.get(camera_id, 0),
            stream_epoch=self._epochs.get(camera_id, 0),
            source_pts=0,
            source_sequence=self._request_sequence,
            native_publish_sequence=0,
            request_id=self._request_sequence,
            transform_id=self._identity.transform_id,
            payload=payload,
        )

    def request(self, message: ControlMessage) -> ControlMessage:
        with self._lock:
            client = self._socket
            if client is None:
                raise ChildControlError("control_closed", str(self._path))
            try:
                client.sendall(encode_message(message))
                raw = client.recv(_MAX_REPLY)
            except OSError as error:
                raise ChildControlError("control_io", str(error)) from error
            if raw == b"":
                raise ChildControlError("control_eof", str(self._path))
            try:
                reply = decode_control_message(raw)
            except IpcProtocolError as error:
                raise ChildControlError("invalid_reply", str(error)) from error
            if reply.worker_boot_id != message.worker_boot_id:
                raise ChildControlError("boot_mismatch", str(reply.worker_boot_id))
            if reply.child_instance_id != message.child_instance_id:
                raise ChildControlError("child_mismatch", str(reply.child_instance_id))
            if reply.request_id != message.request_id:
                raise ChildControlError("correlation_mismatch", str(reply.request_id))
            if reply.kind in _ACCEPTED_REPLIES:
                return reply
            code = "native_error" if reply.kind is MessageKind.ERROR else "reply_kind"
            detail = (
                reply.payload.decode(errors="replace")
                if reply.kind is MessageKind.ERROR
                else reply.kind.name
            )
            raise ChildControlError(code, detail)

    def status(self) -> NativeStatus:
        reply = self.request(self._message(MessageKind.STATUS, "_worker"))
        if len(reply.payload) != _STATUS.size:
            raise ChildControlError("status_size", str(len(reply.payload)))
        payload = reply.payload
        return NativeStatus(
            metadata_published=int.from_bytes(payload[0:8], "little"),
            metadata_overwritten=int.from_bytes(payload[8:16], "little"),
            wake_dropped=int.from_bytes(payload[16:24], "little"),
            source_failures=int.from_bytes(payload[24:32], "little"),
            malformed_frames=int.from_bytes(payload[32:40], "little"),
            source_count=int.from_bytes(payload[40:44], "little"),
            custom_transform_available=bool(payload[44]),
        )

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        self._generations[camera_id] = self._generations.get(camera_id, 0) + 1
        self._epochs[camera_id] = 0
        reply = self.request(self._message(MessageKind.ADD_SOURCE, camera_id, uri.encode()))
        if reply.kind is not MessageKind.EPOCH_STARTED:
            raise ChildControlError("source_not_started", reply.kind.name)
        self._epochs[camera_id] = reply.stream_epoch
        return SourceBinding(
            worker_boot_id=str(self._identity.worker_boot_id),
            child_instance_id=str(self._identity.child_instance_id),
            camera_id=camera_id,
            source_generation=reply.source_generation,
            stream_epoch=reply.stream_epoch,
            transform_id=self._identity.transform_id,
        )

    def remove_source(self, camera_id: str) -> None:
        _ = self.request(self._message(MessageKind.REMOVE_SOURCE, camera_id))
        _ = self._epochs.pop(camera_id, None)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        reply = self.request(
            self._message(MessageKind.SOURCE_FAILURE, camera_id, category.encode())
        )
        if reply.kind is not MessageKind.EPOCH_STARTED:
            raise ChildControlError("epoch_not_started", reply.kind.name)
        self._epochs[camera_id] = reply.stream_epoch
        return SourceBinding(
            worker_boot_id=str(self._identity.worker_boot_id),
            child_instance_id=str(self._identity.child_instance_id),
            camera_id=camera_id,
            source_generation=reply.source_generation,
            stream_epoch=reply.stream_epoch,
            transform_id=self._identity.transform_id,
        )

    def emit_metadata(self, camera_id: str) -> None:
        _ = self.request(self._message(MessageKind.EMIT_METADATA, camera_id))

    def pull_latest(self, camera_id: str) -> MetadataFrame | None:
        try:
            reply = self.request(self._message(MessageKind.GET_LATEST, camera_id))
        except ChildControlError as error:
            raise MetadataPullStopped from error
        if reply.kind is MessageKind.CAPABILITY_INACTIVE:
            return None
        return decode_metadata(encode_message(reply))

    def dark_capability(self, kind: MessageKind, camera_id: str) -> bool:
        if kind not in _DARK_CAPABILITIES:
            raise ChildControlError("not_dark_capability", kind.name)
        reply = self.request(self._message(kind, camera_id))
        return reply.kind is not MessageKind.CAPABILITY_INACTIVE

    def wait_for_publish(self, target: int) -> None:
        payload = target.to_bytes(8, "little")
        _ = self.request(self._message(MessageKind.WAIT_PUBLISH, "_worker", payload))

    def fatal(self, category: str) -> None:
        _ = self.request(self._message(MessageKind.FATAL, "_worker", category.encode()))

    def shutdown(self) -> None:
        _ = self.request(self._message(MessageKind.SHUTDOWN, "_worker"))


__all__ = [
    "ChildControlError",
    "ControlIdentity",
    "DeepStreamControlClient",
    "NativeStatus",
]
