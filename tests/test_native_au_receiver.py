from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field

from worker.adapters.decode.native_au_receiver import NativeAuReceiver
from worker.types.source_packet import SourcePacket, StreamEpoch

_HEADER = struct.Struct("<4sIBBBBIQQqqqiiIIHHII")


# allow: MUTABLE_OK - synchronization fake records receiver effects.
@dataclass(slots=True)
class _Sink:
    packets: list[SourcePacket] = field(default_factory=list)
    epochs: list[StreamEpoch] = field(default_factory=list)
    accepted: threading.Event = field(default_factory=threading.Event)

    def register_camera(self, camera_id: str) -> None:
        assert camera_id == "camera-a"

    def roll_epoch(self, epoch: StreamEpoch) -> None:
        self.epochs.append(epoch)

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        self.accepted.set()
        return True


def _frame(
    *,
    epoch: int,
    sequence: int,
    kind: int = 1,
    payload: bytes = b"\0\0\0\x02e\x80",
) -> bytes:
    camera = b"camera-a"
    caps = b"video/x-h264,alignment=(string)au,stream-format=(string)avc"
    codec_data = b"\x01d\0\x1f\xff"
    body = camera + caps + codec_data + payload
    return (
        _HEADER.pack(
            b"SAU1",
            len(body),
            kind,
            1,
            2,
            1,
            3,
            epoch,
            sequence,
            sequence * 3_000,
            sequence * 3_000,
            3_000,
            1,
            90_000,
            640,
            360,
            len(camera),
            len(caps),
            len(codec_data),
            len(payload),
        )
        + body
    )


def test_native_au_receiver_adapts_parser_facts_into_source_packet() -> None:
    # Given: a dedicated stream channel and the Python packet authority.
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, reason: gaps.append((camera, reason)),
    )
    receiver.start()

    # When: one complete parser-aligned AU arrives.
    child.sendall(_frame(epoch=7, sequence=1))

    # Then: exact epoch/timeline/framing facts reach SourcePacket.
    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    packet = sink.packets[0]
    assert packet.epoch == StreamEpoch("boot-1", "camera-a", 7, 3)
    assert (packet.pts, packet.dts, packet.duration, packet.is_keyframe) == (
        3_000,
        3_000,
        3_000,
        True,
    )
    assert packet.configuration.streams[0].stream_format == "avc"
    assert packet.configuration.streams[0].nal_length_size == 4
    assert not gaps


def test_native_au_receiver_reports_gap_without_appending_across_epoch() -> None:
    # Given: an AU receiver with an exact gap callback subscribed.
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    gaps: list[tuple[str, str]] = []

    def record_gap(camera_id: str, reason: str) -> None:
        gaps.append((camera_id, reason))
        gap_seen.set()

    receiver = NativeAuReceiver(parent, "boot-1", sink, record_gap)
    receiver.start()

    # When: the bounded native sender declares an AU gap.
    child.sendall(_frame(epoch=7, sequence=2, kind=2, payload=b""))

    # Then: no packet is admitted and the source epoch owner is asked to rebuild.
    assert gap_seen.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert gaps == [("camera-a", "parser")]
    assert sink.packets == []
