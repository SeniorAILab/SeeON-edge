from __future__ import annotations

import numpy as np
import pytest

from edge.sources import CameraSource, Frame, FrameSource

# ---------------------------------------------------------------------------
# Fake cv2.VideoCapture
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Minimal cv2.VideoCapture stub driven by a scripted frame list."""

    def __init__(self, frames: list[np.ndarray], *, fail_after: int | None = None) -> None:
        # frames is a list of BGR uint8 arrays; None entries produce read failures.
        self._frames = frames
        self._fail_after = fail_after  # if set, always fail at/after this call index
        self._call_count = 0
        self._released = False

    def set(self, prop: int, value: object) -> None:
        pass  # accept CAP_PROP_BUFFERSIZE without error

    def read(self) -> tuple[bool, np.ndarray | None]:
        idx = self._call_count
        self._call_count += 1
        if self._fail_after is not None and idx >= self._fail_after:
            return False, None
        if idx >= len(self._frames):
            return False, None
        frame = self._frames[idx]
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._released = True


def _bgr_frame(h: int = 4, w: int = 6, fill: int = 128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_camera_source_yields_frame_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [_bgr_frame(fill=i * 30) for i in range(3)]
    cap = _FakeCapture(frames)
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    result = list(CameraSource(0))

    assert len(result) == 3
    assert all(isinstance(f, Frame) for f in result)


def test_camera_source_satisfies_frame_source_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([_bgr_frame()])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    source = CameraSource(0)

    assert isinstance(source, FrameSource)


def test_camera_source_converts_bgr_to_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    # BGR frame with B=10, G=20, R=30 → RGB should be R=30, G=20, B=10
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10  # B
    bgr[:, :, 1] = 20  # G
    bgr[:, :, 2] = 30  # R
    cap = _FakeCapture([bgr])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0))

    rgb = frames[0].image
    assert rgb[0, 0, 0] == 30  # R channel
    assert rgb[0, 0, 1] == 20  # G channel
    assert rgb[0, 0, 2] == 10  # B channel


def test_camera_source_image_is_rgb_hwc_uint8(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([_bgr_frame(h=8, w=12)])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0))

    img = frames[0].image
    assert img.ndim == 3
    assert img.shape == (8, 12, 3)
    assert img.dtype == np.uint8


def test_camera_source_frame_indices_are_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([_bgr_frame() for _ in range(5)])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0))

    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_camera_source_time_sec_is_monotonically_non_decreasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _FakeCapture([_bgr_frame() for _ in range(4)])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0))

    times = [f.time_sec for f in frames]
    assert times[0] == 0.0  # first frame: elapsed from t0 to t0 = 0
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


def test_camera_source_time_sec_first_frame_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([_bgr_frame()])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0))

    assert frames[0].time_sec == 0.0


def test_camera_source_terminates_after_max_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two good frames, then fail_after=2 → always fail → hits max_failures=3 and stops.
    cap = _FakeCapture([_bgr_frame(), _bgr_frame()], fail_after=2)
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0, max_failures=3))

    assert len(frames) == 2


def test_camera_source_resets_failure_counter_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Interleaved failures: two good frames, then two bad, then two good.
    # max_failures=3 → should NOT terminate during the two-bad streak.
    sequence: list[np.ndarray | None] = [
        _bgr_frame(),
        _bgr_frame(),
        None,  # fail
        None,  # fail
        _bgr_frame(),
        _bgr_frame(),
    ]
    cap = _FakeCapture(sequence)
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0, max_failures=3))

    # All four good frames should be yielded despite the two interspersed failures.
    assert len(frames) == 4


def test_camera_source_releases_capture_on_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([_bgr_frame()])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    list(CameraSource(0, max_failures=1))

    assert cap._released


def test_camera_source_can_be_iterated_with_zero_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCapture([])
    monkeypatch.setattr("edge.sources.frame_source.cv2.VideoCapture", lambda _idx: cap)

    frames = list(CameraSource(0, max_failures=1))

    assert frames == []
