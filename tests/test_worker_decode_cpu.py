from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from worker.adapters.decode.cpu_av import CpuAvAdapter, CpuAvConfig
from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.runtime.profile.registry import PROFILE_REGISTRY


class _FakeCapture:
    def __init__(
        self,
        frames: list[tuple[bool, NDArray[np.uint8] | None]],
        *,
        opened: bool = True,
    ) -> None:
        self._frames: list[tuple[bool, NDArray[np.uint8] | None]] = list(frames)
        self._opened: bool = opened
        self.set_calls: list[tuple[int, float]] = []
        self.release_calls: int = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        return self._frames.pop(0) if self._frames else (False, None)

    def release(self) -> None:
        self.release_calls += 1


class _CaptureFactory:
    def __init__(self, captures: list[_FakeCapture]) -> None:
        self._captures: list[_FakeCapture] = list(captures)
        self.calls: list[tuple[str, int | None, list[int] | None]] = []

    def __call__(
        self,
        url: str,
        backend: int | None = None,
        params: list[int] | None = None,
    ) -> _FakeCapture:
        self.calls.append((url, backend, params))
        return self._captures.pop(0)


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _bgr(value: tuple[int, int, int] = (10, 20, 30)) -> NDArray[np.uint8]:
    return np.array([[list(value)]], dtype=np.uint8)


def test_cpu_av_exposes_minimal_public_decode_api() -> None:
    # Given
    expected_symbols = ("CpuAvAdapter", "CpuAvConfig")

    # When
    imported_symbols = (CpuAvAdapter.__name__, CpuAvConfig.__name__)

    # Then
    assert imported_symbols == expected_symbols


def test_cpu_av_opens_with_exact_timeouts_and_capture_properties() -> None:
    # Given
    capture = _FakeCapture([])
    factory = _CaptureFactory([capture])
    adapter = CpuAvAdapter(capture_factory=factory, clock=_Clock([]))
    config = CpuAvConfig(
        camera_id="camera-a",
        url="rtsp://camera/trackID=2",
        open_timeout_ms=1234,
        read_timeout_ms=5678,
    )

    # When
    session = adapter.open(config)

    # Then
    assert factory.calls == [
        (
            "rtsp://camera/trackID=2",
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                1234,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5678,
            ],
        )
    ]
    assert capture.set_calls == [
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5678),
    ]
    session.close()


@pytest.mark.parametrize(
    ("operator_value", "expected"),
    [
        (None, "rtsp_transport;tcp"),
        ("rtsp_transport;udp|max_delay;500000", "rtsp_transport;udp|max_delay;500000"),
    ],
)
def test_cpu_av_defaults_to_tcp_without_overwriting_operator_override(
    monkeypatch: pytest.MonkeyPatch,
    operator_value: str | None,
    expected: str,
) -> None:
    # Given
    if operator_value is None:
        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", operator_value)
    capture = _FakeCapture([])
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([capture]),
        clock=_Clock([]),
    )

    # When
    session = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))

    # Then
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected
    session.close()


def test_cpu_av_emits_immutable_rgb_packets_with_coherent_stream_metadata() -> None:
    # Given
    capture = _FakeCapture(
        [(True, _bgr()), (True, _bgr()), (True, _bgr())],
    )
    clock = _Clock([10.000, 10.001, 10.040, 10.041, 10.080, 10.081])
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([capture]),
        clock=clock,
    )
    session = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))

    # When
    packets = [session.read(), session.read(), session.read()]

    # Then
    assert all(packet is not None for packet in packets)
    first, second, third = packets
    assert first is not None and second is not None and third is not None
    assert first.frame.image.tolist() == [[[30, 20, 10]]]
    assert [packet.camera_id for packet in (first, second, third)] == ["camera-a"] * 3
    assert [(packet.width, packet.height) for packet in (first, second, third)] == [
        (1, 1),
    ] * 3
    assert [packet.seq for packet in (first, second, third)] == [0, 1, 2]
    assert [packet.frame.index for packet in (first, second, third)] == [0, 1, 2]
    assert [round(packet.frame.time_sec, 3) for packet in (first, second, third)] == [
        0.0,
        0.04,
        0.08,
    ]
    assert first.pts is not None and second.pts is not None and third.pts is not None
    assert [round(first.pts, 3), round(second.pts, 3), round(third.pts, 3)] == [
        0.0,
        0.04,
        0.08,
    ]
    assert [round(packet.decode_time_ms, 3) for packet in (first, second, third)] == [
        1.0,
        1.0,
        1.0,
    ]
    assert all(packet.decode_time_ms >= 0 for packet in (first, second, third))
    immutable_field = "seq"
    with pytest.raises(FrozenInstanceError):
        setattr(first, immutable_field, 99)


def test_cpu_av_failed_open_releases_capture() -> None:
    # Given
    capture = _FakeCapture([], opened=False)
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([capture]),
        clock=_Clock([]),
    )

    # When / Then
    with pytest.raises(RuntimeError):
        _ = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))
    assert capture.release_calls == 1


def test_cpu_av_failed_read_closes_capture() -> None:
    # Given
    capture = _FakeCapture([(False, None)])
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([capture]),
        clock=_Clock([1.0, 1.1]),
    )
    session = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))

    # When
    packet = session.read()

    # Then
    assert packet is None
    assert capture.release_calls == 1


def test_cpu_av_close_is_idempotent_and_releases_once() -> None:
    # Given
    capture = _FakeCapture([])
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([capture]),
        clock=_Clock([]),
    )
    session = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))

    # When
    session.close()
    session.close()

    # Then
    assert capture.release_calls == 1


def test_cpu_av_structurally_conforms_to_decode_protocols() -> None:
    # Given
    adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([_FakeCapture([])]),
        clock=_Clock([]),
    )

    # When
    session = adapter.open(CpuAvConfig("camera-a", "rtsp://camera/live"))

    # Then
    assert isinstance(adapter, DecodeAdapter)
    assert isinstance(session, DecodeSession)
    session.close()


def test_profile_policy_selects_cpu_decode_only_for_cpu_and_mps() -> None:
    # Given
    profiles = PROFILE_REGISTRY

    # When
    decode_by_profile = {name: spec.decode for name, spec in profiles.items()}

    # Then
    assert decode_by_profile == {"cuda": "nvdec", "mps": "opencv", "cpu": "opencv"}


def test_rtsp_source_degrades_only_failed_camera_and_closes_its_capture() -> None:
    # Given
    failed_capture = _FakeCapture([(False, None)])
    healthy_capture = _FakeCapture([(True, _bgr())])
    failed_adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([failed_capture]),
        clock=_Clock([1.0, 1.1]),
    )
    healthy_adapter = CpuAvAdapter(
        capture_factory=_CaptureFactory([healthy_capture]),
        clock=_Clock([2.0, 2.1]),
    )
    degraded: list[tuple[str, str]] = []
    failed_source = RTSPSource(
        CpuAvConfig("camera-failed", "rtsp://camera-failed/live"),
        failed_adapter,
        max_failures=1,
        max_total_reconnects=0,
        pace_wait=lambda _delay: True,
    )
    healthy_source = RTSPSource(
        CpuAvConfig("camera-healthy", "rtsp://camera-healthy/live"),
        healthy_adapter,
        max_failures=1,
        max_total_reconnects=0,
        pace_wait=lambda _delay: True,
    )
    failed_source.set_liveness_callbacks(
        on_reconnecting=lambda reason: degraded.append(("camera-failed", reason))
    )
    healthy_source.set_liveness_callbacks(
        on_reconnecting=lambda reason: degraded.append(("camera-healthy", reason))
    )

    # When
    failed_packets = list(failed_source)
    healthy_packets = list(healthy_source)

    # Then
    assert failed_packets == []
    assert [packet.camera_id for packet in healthy_packets] == ["camera-healthy"]
    assert degraded == [("camera-failed", "read_failure")]
    assert failed_capture.release_calls == 1
    assert healthy_capture.release_calls == 1
