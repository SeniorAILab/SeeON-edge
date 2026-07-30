from __future__ import annotations

import os
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from contracts.decode_diagnostics import DecodeSelection
from edge.sources import probe as probe_module
from edge.sources import rtsp as rtsp_module
from edge.sources.probe import (
    RTSPProbeError,
    RTSPProbeResult,
    mask_rtsp_url,
    probe_first_frame,
)
from edge.sources.probe import (
    main as probe_main,
)
from edge.sources.rtsp import RTSPSource, _create_backend, create_backend
from edge.sources.rtsp_backend import (
    FallbackRTSPBackend,
    NvdecRTSPBackend,
    NvdecUnavailableError,
    ObservingRTSPBackend,
    OpenCVRTSPBackend,
    _NvdecCapture,
    _probe_stream_metadata,
)


class _FakeCapture:
    def __init__(self) -> None:
        self._frames = [
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            np.array([[[40, 50, 60]]], dtype=np.uint8),
        ]
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class _RecordingBackend:
    def __init__(self) -> None:
        self.capture = _FakeCapture()
        self.open_calls: list[tuple[str, int, int]] = []
        self.release_calls = 0

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> _FakeCapture:
        self.open_calls.append((url, open_timeout_ms, read_timeout_ms))
        return self.capture

    def read(self, capture: _FakeCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: _FakeCapture) -> None:
        self.release_calls += 1
        capture.release()


class _ProbeCapture:
    def __init__(
        self,
        responses: list[tuple[bool, NDArray[np.uint8] | None]],
    ) -> None:
        self._responses = responses
        self.released = False

    def set(self, prop_id: int, value: float) -> bool:
        del prop_id, value
        return True

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._responses:
            return self._responses.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class _ProbeBackend:
    def __init__(
        self,
        responses: list[tuple[bool, NDArray[np.uint8] | None]] | None = None,
        open_error: Exception | None = None,
    ) -> None:
        self.capture = _ProbeCapture(responses or [])
        self.open_error = open_error
        self.open_calls: list[tuple[str, int, int]] = []
        self.release_calls = 0

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> _ProbeCapture:
        self.open_calls.append((url, open_timeout_ms, read_timeout_ms))
        if self.open_error is not None:
            raise self.open_error
        return self.capture

    def read(self, capture: _ProbeCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: _ProbeCapture) -> None:
        self.release_calls += 1
        capture.release()


class _SequenceCapture:
    def __init__(self, reads: list[tuple[bool, NDArray[np.uint8] | None]]) -> None:
        self._reads = reads
        self.released = False

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if not self._reads:
            return False, None
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True


class _SequenceBackend:
    def __init__(self, reads_by_open: list[list[tuple[bool, NDArray[np.uint8] | None]]]) -> None:
        self._reads_by_open = reads_by_open
        self.captures: list[_SequenceCapture] = []
        self.open_calls = 0
        self.release_calls = 0

    def open(
        self,
        url: str,
        open_timeout_ms: int,
        read_timeout_ms: int,
    ) -> _SequenceCapture:
        del url, open_timeout_ms, read_timeout_ms
        self.open_calls += 1
        reads = self._reads_by_open.pop(0) if self._reads_by_open else []
        capture = _SequenceCapture(reads)
        self.captures.append(capture)
        return capture

    def read(self, capture: _SequenceCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: _SequenceCapture) -> None:
        self.release_calls += 1
        capture.release()

class _OpenOutcomeBackend:
    def __init__(
        self, outcomes: list[Exception | list[tuple[bool, NDArray[np.uint8] | None]]]
    ) -> None:
        self._outcomes = outcomes
        self.open_calls = 0
        self.release_calls = 0

    def open(
        self, url: str, open_timeout_ms: int, read_timeout_ms: int
    ) -> _SequenceCapture:
        del url, open_timeout_ms, read_timeout_ms
        self.open_calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _SequenceCapture(outcome)

    def read(self, capture: _SequenceCapture) -> tuple[bool, NDArray[np.uint8] | None]:
        return capture.read()

    def release(self, capture: _SequenceCapture) -> None:
        self.release_calls += 1
        capture.release()



def test_rtsp_source_uses_injected_rgb_backend_without_double_conversion(monkeypatch) -> None:
    backend = _RecordingBackend()
    monkeypatch.setattr(
        backend.capture,
        "_frames",
        [
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            np.array([[[40, 50, 60]]], dtype=np.uint8),
        ],
    )
    monkeypatch.setattr(cv2, "cvtColor", _fail_if_called)
    monkeypatch.setattr(
        "edge.sources.rtsp.time.monotonic",
        _monotonic_values((10.0, 10.0, 10.25, 10.25)),
    )

    frames = list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            open_timeout_ms=1234,
            read_timeout_ms=5678,
            backend=backend,
            max_total_reconnects=0,
            sleep=lambda _delay: None,
        )
    )

    assert backend.open_calls == [("rtsp://camera/trackID=2", 1234, 5678)]
    assert backend.release_calls == 1
    assert backend.capture.released is True
    assert [frame.index for frame in frames] == [0, 1]
    assert [frame.time_sec for frame in frames] == [0.0, 0.25]
    assert frames[0].image.tolist() == [[[10, 20, 30]]]
    assert frames[1].image.tolist() == [[[40, 50, 60]]]


def test_create_backend_defaults_to_auto_fail_loud_nvdec(monkeypatch) -> None:
    monkeypatch.delenv("ML_RTSP_BACKEND", raising=False)

    # ADR-0002: "auto" is fail-loud NVDEC (no silent OpenCV fallback)
    assert isinstance(create_backend(), NvdecRTSPBackend)



def test_create_backend_selects_opencv_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ML_RTSP_BACKEND", "OPENCV")

    assert isinstance(_create_backend(), OpenCVRTSPBackend)


def test_create_backend_selects_nvdec() -> None:
    assert isinstance(create_backend("nvdec"), NvdecRTSPBackend)


def test_create_backend_cpu_alias_maps_to_opencv() -> None:
    assert isinstance(create_backend("cpu"), OpenCVRTSPBackend)


def test_create_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported RTSP backend"):
        create_backend("vaapi")


class _FailingBackend:
    def open(self, url: str, open_timeout_ms: int, read_timeout_ms: int) -> object:
        raise NvdecUnavailableError("ffmpeg spawn failed: OSError")

    def read(self, capture: object) -> tuple[bool, NDArray[np.uint8] | None]:
        return False, None

    def release(self, capture: object) -> None:
        return None


class _StubBackend:
    def __init__(self, frame: NDArray[np.uint8]) -> None:
        self._frame = frame
        self.opened = False
        self.released = False

    def open(self, url: str, open_timeout_ms: int, read_timeout_ms: int) -> object:
        self.opened = True
        return object()

    def read(self, capture: object) -> tuple[bool, NDArray[np.uint8] | None]:
        return True, self._frame

    def release(self, capture: object) -> None:
        self.released = True


def test_fallback_backend_uses_safe_backend_when_preferred_open_fails() -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    safe = _StubBackend(frame)
    backend = FallbackRTSPBackend([("nvdec", _FailingBackend()), ("opencv", safe)])

    capture = backend.open("rtsp://cam/live", 1000, 1000)
    ok, out = backend.read(capture)

    assert ok is True
    assert out is frame
    assert safe.opened is True


def test_fallback_backend_prefers_first_backend_that_yields_a_frame() -> None:
    preferred_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    preferred = _StubBackend(preferred_frame)
    safe = _StubBackend(np.ones((1, 1, 3), dtype=np.uint8))
    backend = FallbackRTSPBackend([("nvdec", preferred), ("opencv", safe)])

    capture = backend.open("rtsp://cam/live", 1000, 1000)
    ok, out = backend.read(capture)

    assert ok is True
    assert out is preferred_frame  # validating frame is replayed, not lost
    assert safe.opened is False
def test_fallback_backend_records_bounded_reason_and_masks_credentials(caplog) -> None:
    selections: list[DecodeSelection] = []
    safe = _StubBackend(np.zeros((1, 1, 3), dtype=np.uint8))
    backend = FallbackRTSPBackend(
        [
            ("nvdec", _ProbeBackend(open_error=NvdecUnavailableError("ffprobe failed: OSError"))),
            ("opencv", safe),
        ],
        on_selection=selections.append,
        clock=lambda: 12.5,
    )

    capture = backend.open("rtsp://user:password@camera.local/live", 1000, 1000)

    assert capture.backend_name == "opencv"
    assert selections == [
        DecodeSelection(
            requested="auto",
            selected="opencv",
            fallback_count=1,
            last_reason="ffprobe_unavailable",
            updated_at_sec=12.5,
        )
    ]
    assert "password" not in caplog.text
    assert "user" not in caplog.text


def test_observing_backend_records_forced_cpu_selection() -> None:
    selections: list[DecodeSelection] = []
    backend = ObservingRTSPBackend(
        _StubBackend(np.zeros((1, 1, 3), dtype=np.uint8)),
        requested="cpu",
        selected="opencv",
        on_selection=selections.append,
        clock=lambda: 4.0,
    )

    backend.open("rtsp://camera.local/live", 1000, 1000)

    assert selections[0].requested == "cpu"
    assert selections[0].selected == "opencv"


def test_fallback_backend_classifies_first_frame_failure() -> None:
    selections: list[DecodeSelection] = []
    backend = FallbackRTSPBackend(
        [
            ("nvdec", _ProbeBackend([(False, None)])),
            ("opencv", _StubBackend(np.zeros((1, 1, 3), dtype=np.uint8))),
        ],
        on_selection=selections.append,
    )

    backend.open("rtsp://camera.local/live", 1000, 1000)

    assert selections[0].fallback_count == 1
    assert selections[0].last_reason == "first_frame_failed"


def test_probe_first_frame_reports_resolution_channels_and_releases_capture() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    backend = _ProbeBackend([(True, frame)])

    result = probe_first_frame(
        "rtsp://user:secret@camera.local/live?token=abc",
        backend=backend,
        timeout_ms=1000,
        open_timeout_ms=111,
        read_timeout_ms=222,
    )

    assert result.width == 3
    assert result.height == 2
    assert result.channels == 3
    assert result.requested_backend == "injected"
    assert result.backend == "injected"
    assert result.masked_url == "rtsp://***:***@camera.local/live?token=%2A%2A%2A"
    assert backend.open_calls == [
        ("rtsp://user:secret@camera.local/live?token=abc", 111, 222)
    ]
    assert backend.release_calls == 1
    assert backend.capture.released is True


def test_probe_first_frame_auto_fails_loud_when_nvdec_down(monkeypatch) -> None:
    from edge.sources.probe import RTSPProbeError

    monkeypatch.setattr("edge.sources.rtsp.NvdecRTSPBackend", _FailingBackend)

    # ADR-0002: auto no longer silently falls back to OpenCV; NVDEC failure surfaces
    with pytest.raises(RTSPProbeError):
        probe_first_frame("rtsp://camera.local/live", backend_name="auto")


def test_probe_first_frame_reports_forced_nvdec_selection(monkeypatch) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    nvdec = _ProbeBackend([(True, frame)])
    monkeypatch.setitem(rtsp_module._BACKEND_FACTORIES, "nvdec", lambda: nvdec)

    result = probe_first_frame("rtsp://camera.local/live", backend_name="nvdec")

    assert result.requested_backend == "nvdec"
    assert result.backend == "nvdec"


def test_probe_first_frame_reports_cpu_alias_as_selected_opencv(monkeypatch) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    opencv = _ProbeBackend([(True, frame)])
    monkeypatch.setitem(rtsp_module._BACKEND_FACTORIES, "cpu", lambda: opencv)
    monkeypatch.setitem(rtsp_module._BACKEND_FACTORIES, "opencv", lambda: opencv)

    result = probe_first_frame("rtsp://camera.local/live", backend_name="cpu")

    assert result.requested_backend == "cpu"
    assert result.backend == "opencv"


@pytest.mark.parametrize(
    ("argv", "expected_backend"),
    [
        (["rtsp://camera.local/live"], None),
        (["rtsp://camera.local/live", "--backend", "auto"], "auto"),
        (["rtsp://camera.local/live", "--backend", "nvdec"], "nvdec"),
        (["rtsp://camera.local/live", "--backend", "opencv"], "opencv"),
        (["rtsp://camera.local/live", "--backend", "cpu"], "cpu"),
    ],
)
def test_probe_cli_accepts_backend_contract_vocabulary(
    monkeypatch,
    argv: list[str],
    expected_backend: str | None,
) -> None:
    seen_backend_names: list[str | None] = []

    def fake_probe(url: str, **kwargs: object) -> RTSPProbeResult:
        seen_backend_names.append(kwargs["backend_name"])
        return RTSPProbeResult(
            masked_url=url,
            requested_backend="auto",
            backend="opencv",
            width=1,
            height=1,
            channels=3,
        )

    monkeypatch.setattr(probe_module, "probe_first_frame", fake_probe)

    assert probe_main(argv) == 0
    assert seen_backend_names == [expected_backend]


def test_probe_first_frame_classifies_timeout_and_releases_capture() -> None:
    backend = _ProbeBackend([(False, None), (False, None)])
    monotonic = _monotonic_values((0.0, 0.1, 0.2, 0.6))

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame(
            "rtsp://camera.local/live",
            backend=backend,
            timeout_ms=500,
            monotonic=monotonic,
        )

    assert error.value.error_class == "timeout"
    assert "rtsp://camera.local/live" in str(error.value)
    assert backend.release_calls == 1
    assert backend.capture.released is True


def test_probe_first_frame_classifies_decode_failure() -> None:
    backend = _ProbeBackend([(True, None)])

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame("rtsp://camera.local/live", backend=backend)

    assert error.value.error_class == "decode"
    assert "codec" in str(error.value)
    assert backend.release_calls == 1


def test_probe_first_frame_classifies_auth_and_masks_credentials() -> None:
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    backend = _ProbeBackend(open_error=RuntimeError(f"401 Unauthorized for {raw_url}"))

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame(raw_url, backend=backend)

    assert error.value.error_class == "auth"
    assert error.value.masked_url == (
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A"
    )
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)
    assert backend.release_calls == 0


def test_mask_rtsp_url_redacts_userinfo_query_values_and_fragment() -> None:
    assert (
        mask_rtsp_url(
            "rtsp://user:password@host/stream"
            "?profile=main&username=admin&secret=abc#fragment-secret"
        )
        == "rtsp://***:***@host/stream"
        "?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A"
    )


def test_ffprobe_failure_never_exposes_rtsp_credentials(monkeypatch) -> None:
    url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"

    def fail(*_args: object, **_kwargs: object) -> object:
        raise subprocess.CalledProcessError(7, ["ffprobe", url], stderr=url)

    monkeypatch.setattr("edge.sources.rtsp_backend.subprocess.run", fail)

    with pytest.raises(NvdecUnavailableError) as error:
        _probe_stream_metadata(url, 1.0)

    assert str(error.value) == "ffprobe failed: CalledProcessError (returncode=7)"
    assert url not in str(error.value)
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert error.value.__cause__ is None


class _HangingStdout:
    def __init__(self) -> None:
        self._wait = threading.Event()

    def read(self, _size: int) -> bytes:
        self._wait.wait()
        return b""

    def close(self) -> None:
        self._wait.set()


class _HangingProc:
    def __init__(self) -> None:
        self.stdout = _HangingStdout()
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, *, timeout: float) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        self.terminated = True


def test_nvdec_capture_read_times_out_and_cleans_up_hanging_pipe() -> None:
    proc = _HangingProc()
    capture = _NvdecCapture(proc, 1, 1, read_timeout_ms=20)

    started = time.monotonic()
    assert capture.read() == (False, None)

    assert time.monotonic() - started < 0.25
    assert proc.terminated is True
    capture._reader_thread.join(timeout=0.1)
    assert capture._reader_thread.is_alive() is False


def test_nvdec_capture_repeated_timeouts_do_not_accumulate_reader_threads() -> None:
    reader_threads_before = _nvdec_reader_threads()
    capture = _NvdecCapture(_HangingProc(), 1, 1, read_timeout_ms=20)

    assert capture.read() == (False, None)
    assert capture.read() == (False, None)
    assert capture.read() == (False, None)

    capture._reader_thread.join(timeout=0.1)
    assert len(_nvdec_reader_threads()) == len(reader_threads_before)

def test_fallback_backend_propagates_programming_errors() -> None:
    safe = _StubBackend(np.zeros((1, 1, 3), dtype=np.uint8))
    backend = FallbackRTSPBackend(
        [("nvdec", _ProbeBackend(open_error=TypeError("bad backend wiring"))), ("opencv", safe)]
    )

    with pytest.raises(TypeError, match="bad backend wiring"):
        backend.open("rtsp://camera.local/live", 1000, 1000)
    assert safe.opened is False

def test_opencv_rtsp_backend_sets_timeout_and_buffer_properties(monkeypatch) -> None:
    captures: list[_FakeCapture] = []
    calls: list[tuple[str, tuple[object, ...]]] = []

    def video_capture(url: str, *args: object) -> _FakeCapture:
        calls.append((url, args))
        capture = _FakeCapture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", video_capture)

    capture = OpenCVRTSPBackend().open("rtsp://camera/trackID=2", 1234, 5678)

    assert capture is captures[0]
    assert calls == [
        (
            "rtsp://camera/trackID=2",
            (
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    1234,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5678,
                ],
            ),
        )
    ]
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.set_calls
    assert (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5678) in capture.set_calls


def test_opencv_rtsp_backend_defaults_rtsp_transport_to_tcp(monkeypatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: _FakeCapture())

    OpenCVRTSPBackend().open("rtsp://camera/trackID=2", 1234, 5678)

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_opencv_rtsp_backend_preserves_operator_capture_options(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|max_delay;500000"
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: _FakeCapture())

    OpenCVRTSPBackend().open("rtsp://camera/trackID=2", 1234, 5678)

    assert (
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        == "rtsp_transport;udp|max_delay;500000"
    )


def test_opencv_rtsp_backend_read_returns_rgb_frame() -> None:
    capture = _FakeCapture()

    read_ok, image = OpenCVRTSPBackend().read(capture)

    assert read_ok is True
    assert image is not None
    assert image.tolist() == [[[30, 20, 10]]]


def test_opencv_rtsp_backend_releases_capture() -> None:
    capture = _FakeCapture()

    OpenCVRTSPBackend().release(capture)

    assert capture.released is True


def test_opencv_rtsp_backend_falls_back_when_parameterized_open_fails(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def video_capture(url: str, *args: object) -> _FakeCapture:
        calls.append((url, args))
        if len(calls) == 1:
            raise TypeError("three-argument VideoCapture not supported")
        return _FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", video_capture)

    OpenCVRTSPBackend().open("rtsp://camera/trackID=2", 1234, 5678)

    assert calls == [
        (
            "rtsp://camera/trackID=2",
            (
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    1234,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5678,
                ],
            ),
        ),
        ("rtsp://camera/trackID=2", (cv2.CAP_FFMPEG,)),
    ]



def test_rtsp_source_reconnects_after_read_failures_and_resumes(monkeypatch) -> None:
    backend = _SequenceBackend(
        [
            [(False, None), (False, None)],
            [(True, np.array([[[1, 2, 3]]], dtype=np.uint8))],
        ]
    )
    sleeps: list[float] = []
    liveness: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "edge.sources.rtsp.time.monotonic",
        _monotonic_values((100.0,)),
    )
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        max_failures=2,
        reconnect_initial_backoff_sec=0.5,
        reconnect_max_backoff_sec=2.0,
        max_total_reconnects=1,
        sleep=sleeps.append,
        backend=backend,
        pace_wait=lambda _delay: False,
    )
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: liveness.append(("degraded", reason)),
        on_recovered=lambda reason: liveness.append(("ready", reason)),
    )

    frame = next(iter(source))

    assert frame.index == 0
    assert frame.time_sec == 0.0
    assert frame.image.tolist() == [[[1, 2, 3]]]
    assert backend.open_calls == 2
    assert backend.release_calls == 2
    assert sleeps == [0.5]
    assert liveness == [("degraded", "read_failure"), ("ready", "read_recovered")]
def test_rtsp_source_retries_initial_open_failures_before_yielding_a_frame() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    backend = _OpenOutcomeBackend(
        [OSError("unavailable"), OSError("unavailable"), [(True, image)]]
    )
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        backend=backend,
        max_total_reconnects=2,
        sleep=lambda _delay: None,
        pace_wait=lambda _delay: True,
    )

    frame = next(iter(source))

    assert frame.image is image
    assert backend.open_calls == 3

def test_rtsp_source_propagates_programming_errors_from_open() -> None:
    backend = _OpenOutcomeBackend([TypeError("bad backend wiring")])

    with pytest.raises(TypeError, match="bad backend wiring"):
        next(iter(RTSPSource("rtsp://camera/trackID=2", backend=backend)))

    assert backend.open_calls == 1


def test_rtsp_source_recovers_when_reconnect_open_raises() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    backend = _OpenOutcomeBackend(
        [[(False, None)], OSError("reconnect unavailable"), [(True, image)]]
    )
    liveness: list[tuple[str, str]] = []
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        backend=backend,
        max_failures=1,
        max_total_reconnects=2,
        sleep=lambda _delay: None,
        pace_wait=lambda _delay: True,
    )
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: liveness.append(("degraded", reason)),
        on_recovered=lambda reason: liveness.append(("ready", reason)),
    )

    frame = next(iter(source))

    assert frame.image is image
    assert backend.open_calls == 3
    assert liveness == [("degraded", "read_failure"), ("ready", "read_recovered")]


def test_rtsp_source_stop_predicate_cancels_initial_open_backoff() -> None:
    backend = _OpenOutcomeBackend([OSError("unavailable")])
    waits: list[float] = []
    stop = False

    def backoff_wait(delay_sec: float) -> bool:
        nonlocal stop
        waits.append(delay_sec)
        stop = True
        return True

    assert list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            backend=backend,
            backoff_wait=backoff_wait,
            stop_requested=lambda: stop,
            max_total_reconnects=None,
        )
    ) == []
    assert waits == [0.25]
    assert backend.open_calls == 1



def test_rtsp_source_backoff_is_bounded() -> None:
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        reconnect_initial_backoff_sec=0.5,
        reconnect_max_backoff_sec=1.0,
        backend=_SequenceBackend([]),
    )

    assert [source._backoff_delay(reconnect) for reconnect in (1, 2, 3, 4, 10_000)] == [
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_rtsp_source_never_recovered_terminates_with_reconnect_budget() -> None:
    backend = _SequenceBackend(
        [
            [(False, None)],
            [(False, None)],
            [(False, None)],
        ]
    )
    sleeps: list[float] = []

    frames = list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            reconnect_initial_backoff_sec=0.25,
            reconnect_max_backoff_sec=1.0,
            max_total_reconnects=2,
            sleep=sleeps.append,
            backend=backend,
            pace_wait=lambda _delay: False,
        )
    )

    assert frames == []
    assert backend.open_calls == 3
    assert backend.release_calls == 3
    assert sleeps == [0.25, 0.5]


def test_rtsp_source_stop_predicate_cancels_reconnect_backoff_promptly() -> None:
    backend = _SequenceBackend(
        [
            [(False, None)],
        ]
    )
    stop = False
    waits: list[float] = []

    def backoff_wait(delay_sec: float) -> bool:
        nonlocal stop
        waits.append(delay_sec)
        stop = True
        return stop

    frames = list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            reconnect_initial_backoff_sec=30.0,
            reconnect_max_backoff_sec=30.0,
            max_total_reconnects=None,
            backoff_wait=backoff_wait,
            stop_requested=lambda: stop,
            backend=backend,
            pace_wait=lambda _delay: False,
        )
    )

    assert frames == []
    assert backend.open_calls == 1
    assert backend.release_calls == 1
    assert waits == [30.0]


def test_rtsp_source_records_offline_before_reconnect_budget_exhaustion() -> None:
    backend = _SequenceBackend(
        [
            [(False, None)],
            [(False, None)],
        ]
    )
    liveness: list[tuple[str, str]] = []
    source = RTSPSource(
        "rtsp://camera/trackID=2",
        max_failures=1,
        reconnect_initial_backoff_sec=0.25,
        reconnect_max_backoff_sec=1.0,
        max_total_reconnects=1,
        sleep=lambda _delay: None,
        backend=backend,
        pace_wait=lambda _delay: False,
    )
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: liveness.append(("degraded", reason)),
        on_recovered=lambda reason: liveness.append(("ready", reason)),
    )

    assert list(source) == []
    assert backend.open_calls == 2
    assert liveness == [("degraded", "read_failure")]

def test_rtsp_source_paces_processed_fps_with_injected_clock() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    backend = _SequenceBackend([[(True, image), (True, image), (True, image)]])
    now = 10.0
    waits: list[float] = []

    def clock() -> float:
        return now

    def pace_wait(delay_sec: float) -> bool:
        nonlocal now
        waits.append(delay_sec)
        now += delay_sec
        return False

    frames = list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            max_total_reconnects=0,
            target_fps=2.0,
            clock=clock,
            pace_wait=pace_wait,
            backend=backend,
        )
    )

    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.time_sec for frame in frames] == [0.0, 0.5, 1.0]
    assert waits == [0.5, 0.5, 0.5]


def test_rtsp_source_subtracts_consumer_time_from_pacing_delay() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    now = 10.0
    waits: list[float] = []

    def clock() -> float:
        return now

    def pace_wait(delay_sec: float) -> bool:
        nonlocal now
        waits.append(delay_sec)
        now += delay_sec
        return False

    frames = iter(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            max_total_reconnects=0,
            target_fps=2.0,
            clock=clock,
            pace_wait=pace_wait,
            backend=_SequenceBackend([[(True, image), (True, image)]]),
        )
    )

    assert next(frames).time_sec == 0.0
    now += 0.2
    assert next(frames).time_sec == 0.5
    now += 0.6
    assert list(frames) == []
    assert waits == pytest.approx([0.3])


def test_rtsp_source_window_fill_wall_clock_matches_configured_fps() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    window_frames = 30
    fps = 5.0
    now = 100.0

    def clock() -> float:
        return now

    def pace_wait(delay_sec: float) -> bool:
        nonlocal now
        now += delay_sec
        return False

    frames = list(
        RTSPSource(
            "rtsp://camera/trackID=2",
            max_failures=1,
            max_total_reconnects=0,
            target_fps=fps,
            clock=clock,
            pace_wait=pace_wait,
            backend=_SequenceBackend([[(True, image) for _ in range(window_frames)]]),
        )
    )

    assert len(frames) == window_frames
    assert frames[-1].time_sec == (window_frames - 1) / fps

def _nvdec_reader_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "nvdec-stdout-reader"]

def _fail_if_called(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("RTSPSource must not convert backend-provided RGB frames")


def _monotonic_values(values: tuple[float, ...]):
    iterator = iter(values)

    def next_value() -> float:
        return next(iterator)

    return next_value
