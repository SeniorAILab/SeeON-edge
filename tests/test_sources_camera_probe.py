from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from edge.sources.camera_probe import CameraInfo, probe_cameras

# ---------------------------------------------------------------------------
# Fake cv2.VideoCapture for probe tests
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Stub that returns one frame for allowed indices and fails for others."""

    def __init__(self, index: int, open_indices: set[int]) -> None:
        self._index = index
        self._open = index in open_indices
        self._read_called = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._open or self._read_called:
            return False, None
        self._read_called = True
        # Return a distinct BGR frame so we can verify the thumbnail channel order.
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = self._index * 10  # B varies by index
        return True, frame

    def release(self) -> None:
        pass


def _make_factory(open_indices: set[int]):
    def _factory(idx: int) -> _FakeCapture:
        return _FakeCapture(idx, open_indices)

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_probe_cameras_returns_only_open_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _make_factory({0, 2}))

    cameras = probe_cameras(max_index=4)

    assert [c.index for c in cameras] == [0, 2]


def test_probe_cameras_returns_camera_info_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _make_factory({1}))

    cameras = probe_cameras(max_index=3)

    assert len(cameras) == 1
    assert isinstance(cameras[0], CameraInfo)


def test_probe_cameras_thumbnail_is_rgb_hwc_uint8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _make_factory({0}))

    cameras = probe_cameras(max_index=1)

    thumb = cameras[0].thumbnail
    assert thumb.ndim == 3
    assert thumb.shape[2] == 3
    assert thumb.dtype == np.uint8


def test_probe_cameras_thumbnail_bgr_converted_to_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    # Index 0 → B channel = 0, G = 0, R = 0.  The fake returns B=0*10=0.
    # For index 1 → B channel = 1*10=10 in the BGR frame.
    # After BGR→RGB conversion, the first channel (R in output) should equal the
    # last channel of the BGR input (which was R=0 for both) and the B input
    # should move to the last channel.
    class _SingleCapture:
        def __init__(self, idx: int) -> None:
            self._ok = idx == 0

        def read(self):
            if not self._ok:
                return False, None
            bgr = np.zeros((4, 6, 3), dtype=np.uint8)
            bgr[:, :, 0] = 50  # B
            bgr[:, :, 1] = 100  # G
            bgr[:, :, 2] = 150  # R
            return True, bgr

        def release(self):
            pass

    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _SingleCapture)

    cameras = probe_cameras(max_index=1)

    thumb = cameras[0].thumbnail
    assert thumb[0, 0, 0] == 150  # R (was cv2 channel 2)
    assert thumb[0, 0, 1] == 100  # G (was cv2 channel 1)
    assert thumb[0, 0, 2] == 50  # B (was cv2 channel 0)


def test_probe_cameras_empty_when_none_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _make_factory(set()))

    cameras = probe_cameras(max_index=5)

    assert cameras == []


def test_probe_cameras_respects_max_index(monkeypatch: pytest.MonkeyPatch) -> None:
    # All indices up to 10 are "open", but max_index=3 should only probe 0,1,2.
    call_log: list[int] = []

    class _LogCapture:
        def __init__(self, idx: int) -> None:
            call_log.append(idx)
            self._idx = idx

        def read(self):
            return True, np.zeros((4, 6, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _LogCapture)

    probe_cameras(max_index=3)

    assert call_log == [0, 1, 2]


def test_probe_cameras_camera_info_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edge.sources.camera_probe.cv2.VideoCapture", _make_factory({0}))

    cameras = probe_cameras(max_index=1)

    with pytest.raises(FrozenInstanceError):
        cameras[0].index = 99  # type: ignore[misc]
