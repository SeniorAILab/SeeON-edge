"""Versioned binary IPC for the dark DeepStream child."""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, override

from worker.types.perception_frame import (
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBoxChannel,
)

_MAGIC: Final = b"SDS1"
PROTOCOL_VERSION: Final = 1
_HEADER: Final = struct.Struct("<4sBBHIIQQQQQ16s16sHH")
_CHANNELS: Final = struct.Struct("<BBBB")
_MAX_FRAME: Final = 65_535


class MessageKind(IntEnum):
    ADD_SOURCE = 1
    REMOVE_SOURCE = 2
    RECORD = 3
    SNAPSHOT = 4
    STATUS = 5
    SOURCE_FAILURE = 6
    FATAL = 7
    EMIT_METADATA = 8
    GET_LATEST = 9
    SHUTDOWN = 10
    WAIT_PUBLISH = 11
    ACK = 64
    STATUS_REPLY = 65
    ERROR = 66
    EPOCH_STARTED = 67
    CAPABILITY_INACTIVE = 68
    METADATA = 128


@dataclass(frozen=True, slots=True)
class IpcProtocolError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ControlMessage:
    kind: MessageKind
    worker_boot_id: uuid.UUID
    child_instance_id: uuid.UUID
    camera_id: str
    source_generation: int
    stream_epoch: int
    source_pts: int
    source_sequence: int
    native_publish_sequence: int
    request_id: int
    transform_id: str
    payload: bytes = b""


@dataclass(frozen=True, slots=True)
class MetadataFrame:
    frame: PerceptionFrameV1
    source_generation: int
    child_instance_id: uuid.UUID
    native_publish_sequence: int
    transform_id: str

    @property
    def identity(self) -> PerceptionFrameIdentity:
        return self.frame.identity

    @classmethod
    def empty(cls, envelope: ControlMessage) -> MetadataFrame:
        identity = PerceptionFrameIdentity(
            worker_boot_id=str(envelope.worker_boot_id),
            camera_id=envelope.camera_id,
            stream_epoch=envelope.stream_epoch,
            seq=envelope.source_sequence,
            source_pts=envelope.source_pts,
        )
        empty = ChannelState.INFERRED_EMPTY
        return cls(
            frame=PerceptionFrameV1(
                identity=identity,
                person_box=PersonBoxChannel(empty),
                human_pose=HumanPoseChannel(empty),
                bed_region=BedRegionChannel(empty),
            ),
            source_generation=envelope.source_generation,
            child_instance_id=envelope.child_instance_id,
            native_publish_sequence=envelope.native_publish_sequence,
            transform_id=envelope.transform_id,
        )


@dataclass(frozen=True, slots=True)
class MetadataCounters:
    accepted: int = 0
    overwritten: int = 0
    late: int = 0
    unknown_source: int = 0
    generation_mismatch: int = 0
    epoch_mismatch: int = 0
    boot_mismatch: int = 0
    child_mismatch: int = 0
    transform_mismatch: int = 0
    malformed: int = 0


def encode_message(message: ControlMessage) -> bytes:
    camera = message.camera_id.encode()
    transform = message.transform_id.encode()
    body = camera + transform + message.payload
    if (
        len(camera) > 65_535
        or len(transform) > 65_535
        or _HEADER.size + len(body) > _MAX_FRAME
    ):
        raise IpcProtocolError("frame_too_large", str(_HEADER.size + len(body)))
    return _HEADER.pack(
        _MAGIC,
        PROTOCOL_VERSION,
        int(message.kind),
        0,
        len(body),
        message.source_generation,
        message.stream_epoch,
        message.source_pts,
        message.source_sequence,
        message.native_publish_sequence,
        message.request_id,
        message.worker_boot_id.bytes,
        message.child_instance_id.bytes,
        len(camera),
        len(transform),
    ) + body


def decode_control_message(data: bytes) -> ControlMessage:
    if len(data) < _HEADER.size:
        raise IpcProtocolError("short_header", str(len(data)))
    magic = data[0:4]
    version, kind_value = data[4], data[5]
    body_size = int.from_bytes(data[8:12], "little")
    generation = int.from_bytes(data[12:16], "little")
    epoch = int.from_bytes(data[16:24], "little")
    pts = int.from_bytes(data[24:32], "little")
    sequence = int.from_bytes(data[32:40], "little")
    publish_sequence = int.from_bytes(data[40:48], "little")
    request_id = int.from_bytes(data[48:56], "little")
    boot_bytes, child_bytes = data[56:72], data[72:88]
    camera_size = int.from_bytes(data[88:90], "little")
    transform_size = int.from_bytes(data[90:92], "little")
    if magic != _MAGIC or version != PROTOCOL_VERSION:
        raise IpcProtocolError("protocol_mismatch", f"magic={magic!r} version={version}")
    if len(data) != _HEADER.size + body_size:
        raise IpcProtocolError("framing_mismatch", f"declared={body_size} actual={len(data)}")
    try:
        kind = MessageKind(kind_value)
    except ValueError as error:
        raise IpcProtocolError("unknown_kind", str(kind_value)) from error
    if camera_size + transform_size > body_size:
        raise IpcProtocolError("identity_overflow", str(body_size))
    body = memoryview(data)[_HEADER.size:]
    try:
        camera = bytes(body[:camera_size]).decode()
        transform = bytes(body[camera_size : camera_size + transform_size]).decode()
    except UnicodeDecodeError as error:
        raise IpcProtocolError("identity_encoding", str(error)) from error
    payload = bytes(body[camera_size + transform_size :])
    return ControlMessage(
        kind=kind,
        worker_boot_id=uuid.UUID(bytes=boot_bytes),
        child_instance_id=uuid.UUID(bytes=child_bytes),
        camera_id=camera,
        source_generation=generation,
        stream_epoch=epoch,
        source_pts=pts,
        source_sequence=sequence,
        native_publish_sequence=publish_sequence,
        request_id=request_id,
        transform_id=transform,
        payload=payload,
    )


def _encode_state(state: ChannelState) -> int:
    return {
        ChannelState.INFERRED: 1,
        ChannelState.INFERRED_EMPTY: 2,
        ChannelState.SKIPPED: 3,
    }[state]


def encode_metadata(metadata: MetadataFrame) -> bytes:
    frame = metadata.frame
    payload = _CHANNELS.pack(
        _encode_state(frame.person_box.state),
        _encode_state(frame.human_pose.state),
        _encode_state(frame.bed_region.state),
        1 if frame.association is not None else 0,
    )
    return encode_message(
        ControlMessage(
            kind=MessageKind.METADATA,
            worker_boot_id=uuid.UUID(frame.identity.worker_boot_id),
            child_instance_id=metadata.child_instance_id,
            camera_id=frame.identity.camera_id,
            source_generation=metadata.source_generation,
            stream_epoch=frame.identity.stream_epoch,
            source_pts=frame.identity.source_pts or 0,
            source_sequence=frame.identity.seq,
            native_publish_sequence=metadata.native_publish_sequence,
            request_id=0,
            transform_id=metadata.transform_id,
            payload=payload,
        )
    )


def decode_metadata(data: bytes) -> MetadataFrame:
    message = decode_control_message(data)
    if message.kind is not MessageKind.METADATA or message.payload != _CHANNELS.pack(2, 2, 2, 0):
        raise IpcProtocolError("unsupported_metadata", str(len(message.payload)))
    return MetadataFrame.empty(message)


__all__ = [
    "ControlMessage",
    "IpcProtocolError",
    "MessageKind",
    "MetadataCounters",
    "MetadataFrame",
    "PROTOCOL_VERSION",
    "decode_control_message",
    "decode_metadata",
    "encode_message",
    "encode_metadata",
]
