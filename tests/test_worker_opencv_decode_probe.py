from __future__ import annotations

from typing import Any

import pytest

from worker.adapters.decode.cpu_av.probe import OpenCvCapability, probe_opencv_ffmpeg_capability


def test_opencv_capability_true_when_cv2_imports() -> None:
    # Given
    fake_cv2 = object()

    # When
    capability = probe_opencv_ffmpeg_capability(importer=lambda: fake_cv2)

    # Then -- matches edge/runtime/profile/registry.py:151-157's ported semantics:
    # import success alone is sufficient, no further backend introspection.
    assert capability == OpenCvCapability(True, "OpenCV is available")


def test_opencv_capability_false_when_cv2_import_fails() -> None:
    def failing_importer() -> Any:
        raise ImportError("no module named cv2")

    # When
    capability = probe_opencv_ffmpeg_capability(importer=failing_importer)

    # Then -- fail closed, never raises past the probe boundary
    assert capability.available is False
    assert capability.reason == "no module named cv2"


def test_opencv_capability_false_when_cv2_import_raises_non_import_error() -> None:
    # Given -- a broken native shared library can raise OSError rather than
    # ImportError; the ported reference (edge/runtime/profile/registry.py:155)
    # catches Exception broadly, not just ImportError.
    def failing_importer() -> Any:
        raise OSError("dlopen failed: libopencv_videoio.so.4 not found")

    # When
    capability = probe_opencv_ffmpeg_capability(importer=failing_importer)

    # Then
    assert capability.available is False
    assert "dlopen failed" in capability.reason


def test_probe_opencv_ffmpeg_capability_against_real_cv2() -> None:
    """No fakes: exercises the real ``import cv2`` default on this host.

    This is the same signal the worker actually boots against, so it is worth
    running for real (not just via the injected fake) wherever cv2 is
    installed -- which is always, in this repo's test environment. Ground
    truth for this host: cv2 imports cleanly (4.13.0, FFMPEG-backed) and
    separately-verified real cv2.VideoCapture calls against a live RTSP
    stream do open and read real frames, so this probe must report True here.
    """
    pytest.importorskip("cv2")

    capability = probe_opencv_ffmpeg_capability()

    assert capability == OpenCvCapability(True, "OpenCV is available")
