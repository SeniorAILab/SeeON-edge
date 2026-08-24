from __future__ import annotations

import hashlib
import socket
import struct
import threading
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Final, Protocol, final

from worker.adapters.decode.native_au_mux_template import (
    NativeAuTemplateInput,
    build_native_au_mux_template,
    native_configuration_signature,
)
from worker.adapters.decode.native_au_progress import NativeAuProgress
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

_HEADER: Final = struct.Struct("<4sIBBBBIQQqqqiiIIHHII")
_MAGIC: Final = b"SAU1"
_MAX_FRAME: Final = 32 * 1024 * 1024


class _Kind(IntEnum):
    ACCESS_UNIT = 1
    GAP = 2


class _Codec(IntEnum):
    H264 = 1
    H265 = 2


class _Framing(IntEnum):
    ANNEX_B = 1
    AVCC = 2


class NativeAuSink(Protocol):
    def register_camera(self, camera_id: str) -> None: ...
    def append(self, packet: SourcePacket) -> bool: ...
    def roll_epoch(self, epoch: StreamEpoch) -> None: ...


class NativeAuGapHandler(Protocol):
    def __call__(self, camera_id: str, category: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _Envelope:
    kind: _Kind
    codec: _Codec
    framing: _Framing
    keyframe: bool
    generation: int
    epoch: int
    sequence: int
    pts: int
    dts: int
    duration: int
    time_base: Fraction
    width: int
    height: int
    camera_id: str
    parser_caps: str
    codec_data: bytes
    payload: bytes


@final
class NativeAuReceiver:
    """Drain lossless AU IPC; any declared/observed gap rebuilds the native source."""

    def __init__(
        self,
        endpoint: socket.socket,
        worker_boot_id: str,
        sink: NativeAuSink,
        gap_handler: NativeAuGapHandler,
    ) -> None:
        self._endpoint = endpoint
        self._worker_boot_id = worker_boot_id
        self._sink = sink
        self._gap_handler = gap_handler
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._epochs: dict[str, StreamEpoch] = {}
        self._sequences: dict[tuple[str, int, int], int] = {}
        self._configurations: dict[tuple[str, int, int], SourceStreamConfiguration] = {}
        self._configuration_signatures: dict[tuple[str, int, int], str] = {}
        self._progress = NativeAuProgress()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="deepstream-au-receiver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with suppress(OSError):
            self._endpoint.shutdown(socket.SHUT_RDWR)
        self._endpoint.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def accepted_count(self, camera_id: str) -> int:
        return self._progress.count(camera_id)

    def wait_for_packets(self, camera_id: str, target: int, timeout: float) -> bool:
        return self._progress.wait(camera_id, target, timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                header = _recv_exact(self._endpoint, _HEADER.size)
                envelope = _decode(header, _recv_exact(self._endpoint, _body_size(header)))
            except (ConnectionError, OSError, ValueError):
                return
            self._accept(envelope)

    def _accept(self, envelope: _Envelope) -> None:
        identity = StreamEpoch(
            self._worker_boot_id,
            envelope.camera_id,
            envelope.epoch,
            envelope.generation,
        )
        key = (envelope.camera_id, envelope.generation, envelope.epoch)
        expected = self._sequences.get(key, 0) + 1
        if envelope.kind is _Kind.GAP or envelope.sequence != expected:
            self._gap_handler(envelope.camera_id, "parser")
            return
        self._sequences[key] = envelope.sequence
        active = self._epochs.get(envelope.camera_id)
        if active != identity:
            self._sink.register_camera(envelope.camera_id)
            self._sink.roll_epoch(identity)
            self._epochs[envelope.camera_id] = identity
        signature = native_configuration_signature(
            envelope.codec, envelope.framing, envelope.parser_caps, envelope.codec_data,
            envelope.width, envelope.height, envelope.time_base,
        )
        configuration = self._configurations.get(key)
        if configuration is None:
            configuration = _configuration(envelope)
            self._configurations[key] = configuration
            self._configuration_signatures[key] = signature
        elif self._configuration_signatures[key] != signature:
            self._gap_handler(envelope.camera_id, "caps")
            return
        if not self._sink.append(
            SourcePacket(
                identity,
                configuration,
                0,
                envelope.pts,
                envelope.dts,
                envelope.duration,
                envelope.keyframe,
                envelope.payload,
                envelope.sequence - 1,
            )
        ):
            self._gap_handler(envelope.camera_id, "parser")
            return
        self._progress.accept(envelope.camera_id)


def _recv_exact(endpoint: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = endpoint.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("native AU stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _body_size(header: bytes) -> int:
    if len(header) != _HEADER.size or header[:4] != _MAGIC:
        raise ValueError("native AU header invalid")
    size = int.from_bytes(header[4:8], "little")
    if size > _MAX_FRAME:
        raise ValueError("native AU frame exceeds bound")
    return size


def _decode(header: bytes, body: bytes) -> _Envelope:
    try:
        kind = _Kind(header[8])
        codec = _Codec(header[9])
        framing = _Framing(header[10])
    except ValueError as error:
        raise ValueError("native AU variant invalid") from error
    camera_size = int.from_bytes(header[72:74], "little")
    caps_size = int.from_bytes(header[74:76], "little")
    codec_size = int.from_bytes(header[76:80], "little")
    payload_size = int.from_bytes(header[80:84], "little")
    if camera_size + caps_size + codec_size + payload_size != len(body):
        raise ValueError("native AU body framing invalid")
    camera_end = camera_size
    caps_end = camera_end + caps_size
    codec_end = caps_end + codec_size
    return _Envelope(
        kind,
        codec,
        framing,
        bool(header[11]),
        int.from_bytes(header[12:16], "little"),
        int.from_bytes(header[16:24], "little"),
        int.from_bytes(header[24:32], "little"),
        int.from_bytes(header[32:40], "little", signed=True),
        int.from_bytes(header[40:48], "little", signed=True),
        int.from_bytes(header[48:56], "little", signed=True),
        Fraction(
            int.from_bytes(header[56:60], "little", signed=True),
            int.from_bytes(header[60:64], "little", signed=True),
        ),
        int.from_bytes(header[64:68], "little"),
        int.from_bytes(header[68:72], "little"),
        body[:camera_end].decode(),
        body[camera_end:caps_end].decode(),
        body[caps_end:codec_end],
        body[codec_end:],
    )


def _configuration(envelope: _Envelope) -> SourceStreamConfiguration:
    match envelope.codec:
        case _Codec.H264:
            codec_name = "h264"
        case _Codec.H265:
            codec_name = "hevc"
    match envelope.framing:
        case _Framing.ANNEX_B:
            stream_format = "byte-stream"
            # The Python MP4 normalizer writes four-byte lengths and records
            # that choice. Annex-B has no lengthSizeMinusOne field to inherit.
            nal_length_size = 4
        case _Framing.AVCC:
            stream_format = "avc" if codec_name == "h264" else "hvc1"
            nal_length_size = _nal_length_size(envelope.codec_data, codec_name)
    descriptor = SourceStreamDescriptor(
        0,
        "video",
        codec_name,
        "avc1" if codec_name == "h264" else "hvc1",
        envelope.time_base,
        envelope.codec_data,
        envelope.width,
        envelope.height,
        stream_format=stream_format,
        alignment="au",
        nal_length_size=nal_length_size,
        parser_caps_sha256=hashlib.sha256(envelope.parser_caps.encode()).hexdigest(),
    )
    return SourceStreamConfiguration.from_streams(
        (descriptor,),
        mux_template=build_native_au_mux_template(
            descriptor,
            NativeAuTemplateInput(
                envelope.payload,
                envelope.duration,
                envelope.keyframe,
            ),
        ),
    )


def _nal_length_size(codec_data: bytes, codec_name: str) -> int:
    index = 4 if codec_name == "h264" else 21
    if len(codec_data) <= index:
        raise ValueError("native codec data lacks NAL length size")
    return (codec_data[index] & 0b11) + 1


__all__ = ["NativeAuGapHandler", "NativeAuReceiver", "NativeAuSink"]
