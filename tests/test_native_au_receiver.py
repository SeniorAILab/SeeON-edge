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
    generation: int = 3,
    width: int = 640,
    height: int = 360,
    time_base_denominator: int = 90_000,
    pts: int | None = None,
    framing: int = 2,
    codec_data: bytes = b"\x01d\0\x1f\xff",
    caps_suffix: bytes = b"",
) -> bytes:
    camera = b"camera-a"
    stream_format = b"avc" if framing == 2 else b"byte-stream"
    caps = b"video/x-h264,alignment=(string)au,stream-format=(string)" + stream_format + caps_suffix
    body = camera + caps + codec_data + payload
    return (
        _HEADER.pack(
            b"SAU1",
            len(body),
            kind,
            1,
            framing,
            1,
            generation,
            epoch,
            sequence,
            sequence * 3_000 if pts is None else pts,
            sequence * 3_000 if pts is None else pts,
            3_000,
            1,
            time_base_denominator,
            width,
            height,
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
    accepted: list[tuple[str, int, int, int]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, reason: gaps.append((camera, reason)),
        accept_handler=lambda camera, pts, sequence, generation: accepted.append(
            (camera, pts, sequence, generation)
        ),
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
    assert accepted == [("camera-a", 3_000, 1, 3)]


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


def test_malformed_camera_au_does_not_kill_other_camera_drain() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    gaps: list[tuple[str, str]] = []

    def gap(camera_id: str, category: str) -> None:
        gaps.append((camera_id, category))
        gap_seen.set()

    receiver = NativeAuReceiver(parent, "boot-1", sink, gap)
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1, width=0))
    assert gap_seen.wait(timeout=2.0)
    child.sendall(_frame(epoch=8, sequence=1))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert gaps == [("camera-a", "parser")]
    assert sink.packets[-1].epoch.stream_epoch == 8


def test_backward_epoch_is_dropped_without_recovery_but_higher_generation_readd_survives() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(parent, "boot-1", sink, lambda *gap: gaps.append(gap))
    receiver.start()
    child.sendall(_frame(epoch=2, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=1, sequence=1))
    child.sendall(_frame(epoch=1, sequence=1, generation=4))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    # CONTRACT CHANGE, deliberate and evidence-backed. This previously asserted
    # gaps == [("camera-a", "parser")]: a backward epoch asked for a source
    # rebuild. On the live 13-camera fleet that rule was self-sustaining --
    # access units already in flight when the epoch rolled arrive carrying the
    # superseded identity, each one requests a rebuild, each rebuild advances
    # the epoch and strands the next batch. Measured: 301 rebuilds in five
    # minutes with ZERO child-reported failures and no worker restart.
    #
    # A superseded epoch is now dropped the same way a retired generation
    # already was, a few lines above it in _accept. The unit is still rejected;
    # what changed is that rejection no longer triggers recovery.
    #
    # The other half of this test is unchanged and still load-bearing: a higher
    # generation re-add carrying a lower epoch must still be accepted.
    assert gaps == []
    assert sink.packets[-1].epoch == StreamEpoch("boot-1", "camera-a", 1, 4)


def test_timestamp_jump_requests_epoch_roll_without_appending_jump() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda _camera, _category: gap_seen.set(),
    )
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=7, sequence=2, pts=900_000))

    assert gap_seen.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert len(sink.packets) == 1


def test_caps_and_framing_changes_fail_closed_without_splicing() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    gaps: list[tuple[str, str]] = []

    def gap(camera: str, category: str) -> None:
        gaps.append((camera, category))
        gap_seen.set()

    receiver = NativeAuReceiver(parent, "boot-1", sink, gap)
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=7, sequence=2, caps_suffix=b",profile=(string)high"))
    assert gap_seen.wait(timeout=2.0)
    gap_seen.clear()
    child.sendall(
        _frame(
            epoch=8,
            sequence=1,
            framing=1,
            codec_data=b"",
            payload=b"\0\0\0\1\x65\x80",
        )
    )
    assert sink.accepted.wait(timeout=2.0)

    receiver.close()
    child.close()
    assert gaps == [("camera-a", "parser")]
    assert [packet.epoch.stream_epoch for packet in sink.packets] == [7, 8]


def test_short_avcc_codec_data_isolated_then_next_epoch_survives() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda _camera, _category: gap_seen.set()
    )
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1, codec_data=b"\x01"))
    assert gap_seen.wait(timeout=2.0)
    child.sendall(_frame(epoch=8, sequence=1))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert sink.packets[-1].epoch.stream_epoch == 8


def test_in_flight_units_from_a_rolled_epoch_do_not_request_another_rebuild() -> None:
    """Regression for the self-sustaining rebuild storm (#424).

    Reporting a gap for every superseded-epoch unit made recovery feed itself:
    each rebuild advanced the epoch, stranding the units already in flight,
    which requested another rebuild. Measured on the live fleet at 301 rebuilds
    in five minutes with zero child-reported failures.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, category: gaps.append((camera, category)),
    )
    receiver.start()
    # Given: the receiver has adopted epoch 5
    child.sendall(_frame(epoch=5, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    accepted_before = len(sink.packets)
    # When: several units that were in flight during the roll arrive late
    for sequence in range(1, 4):
        child.sendall(_frame(epoch=4, sequence=sequence))
    child.sendall(_frame(epoch=5, sequence=2))
    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    # Then: none of them asked for recovery, and none of them entered the ring
    assert gaps == []
    assert len(sink.packets) == accepted_before + 1
    assert all(packet.epoch.stream_epoch == 5 for packet in sink.packets)
