from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from edge.runtime import mjpeg_server
from edge.runtime.mjpeg_server import (
    MjpegServer,
    OverlayFrameBuffer,
    dev_mjpeg_enabled,
    dev_mjpeg_host,
)
from edge.sources.probe import RTSPProbeError, RTSPProbeResult


def test_mjpeg_buffer_is_camera_keyed_non_consuming() -> None:
    buffer = OverlayFrameBuffer()
    buffer.register_camera("camera-b")
    buffer.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)
    first = buffer.get_latest("camera-a")
    second = buffer.get_latest("camera-a")
    assert first is not None
    assert second is not None
    assert first.jpeg == b"jpeg-1"
    assert second.jpeg == b"jpeg-1"
    assert buffer.get_latest("camera-b") is None

    buffer.publish_jpeg("camera-a", b"jpeg-2", frame_index=2)
    assert buffer.get_latest("camera-a") is not None
    assert buffer.get_latest("camera-a").jpeg == b"jpeg-2"
    assert buffer.get_latest("camera-b") is None


def test_mjpeg_server_defaults_loopback_and_disabled() -> None:
    assert dev_mjpeg_enabled({}) is False
    assert dev_mjpeg_host({}) == "127.0.0.1"
    buffer = OverlayFrameBuffer()
    server = MjpegServer(buffer, port=0)
    try:
        assert server.host == "127.0.0.1"
    finally:
        server.stop()


def test_mjpeg_probe_response_keeps_selected_backend(monkeypatch) -> None:
    expected = RTSPProbeResult(
        masked_url="rtsp://camera.local/live",
        requested_backend="auto",
        backend="opencv",
        width=640,
        height=360,
        channels=3,
    )
    monkeypatch.setattr(mjpeg_server, "probe_first_frame", lambda _url: expected)

    assert mjpeg_server._probe_rtsp_first_frame("rtsp://camera.local/live") == {
        "ok": True,
        "backend": "opencv",
        "width": 640,
        "height": 360,
    }


def test_mjpeg_server_unknown_empty_and_stream_response() -> None:
    buffer = OverlayFrameBuffer()
    buffer.register_camera("empty")
    buffer.publish_jpeg("camera-a", b"\xff\xd8jpeg\xff\xd9", frame_index=1)
    server = MjpegServer(buffer, port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/stream/missing", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(f"{base}/stream/empty", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(f"{base}/stream/camera-a", timeout=1) as response:
            body = response.read(64)
            assert response.status == 200
            assert b"multipart" in response.headers["Content-Type"].encode()
            assert b"\xff\xd8jpeg" in body
    finally:
        server.stop()


def test_snapshot_unknown_empty_and_happy_path() -> None:
    buffer = OverlayFrameBuffer()
    buffer.register_camera("empty")
    buffer.publish_jpeg("camera-a", b"\xff\xd8jpeg\xff\xd9", frame_index=1)
    server = MjpegServer(buffer, port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/snapshot/missing", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(f"{base}/snapshot/empty", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(f"{base}/snapshot/camera-a", timeout=1) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert body == b"\xff\xd8jpeg\xff\xd9"
    finally:
        server.stop()


def test_mjpeg_server_probe_requires_token_and_returns_sanitized_result() -> None:
    buffer = OverlayFrameBuffer()
    seen_urls: list[str] = []

    def probe(url: str) -> dict[str, object]:
        seen_urls.append(url)
        return {"ok": True, "backend": "opencv", "width": 640, "height": 360}

    server = MjpegServer(buffer, port=0, probe_token="relay-token", probe=probe)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    body = json.dumps({"rtsp_url": "rtsp://user:secret@camera.local/trackID=2"}).encode()
    try:
        request = urllib.request.Request(f"{base}/probe", data=body, method="POST")
        try:
            urllib.request.urlopen(request, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover
            raise AssertionError("probe should require relay token")

        request = urllib.request.Request(
            f"{base}/probe",
            data=body,
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"backend": "opencv", "height": 360, "ok": True, "width": 640}
        assert seen_urls == ["rtsp://user:secret@camera.local/trackID=2"]
        assert "secret" not in json.dumps(payload)
        assert "camera.local" not in json.dumps(payload)
    finally:
        server.stop()


def test_mjpeg_server_probe_normalizes_auth_failure_without_leaking_url() -> None:
    buffer = OverlayFrameBuffer()

    def probe(url: str) -> dict[str, object]:
        raise RTSPProbeError("auth", f"auth failed for {url}", "rtsp://***:***@camera/track")

    server = MjpegServer(buffer, port=0, probe_token="relay-token", probe=probe)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    body = json.dumps({"rtsp_url": "rtsp://user:secret@camera.local/trackID=2"}).encode()
    try:
        request = urllib.request.Request(
            f"{base}/probe",
            data=body,
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"error_class": "auth", "ok": False}
        assert "secret" not in json.dumps(payload)
        assert "camera.local" not in json.dumps(payload)
    finally:
        server.stop()


def test_mjpeg_stream_emits_multiple_camera_keyed_non_consuming_parts() -> None:
    buffer = OverlayFrameBuffer()
    buffer.publish_jpeg("camera-a", b"\xff\xd8camera-a-1\xff\xd9", frame_index=1)
    buffer.publish_jpeg("camera-b", b"\xff\xd8camera-b-1\xff\xd9", frame_index=1)
    server = MjpegServer(buffer, port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"

    def publish_updates() -> None:
        time.sleep(0.05)
        buffer.publish_jpeg("camera-b", b"\xff\xd8camera-b-2\xff\xd9", frame_index=2)
        buffer.publish_jpeg("camera-a", b"\xff\xd8camera-a-2\xff\xd9", frame_index=2)
        time.sleep(0.05)
        buffer.publish_jpeg("camera-a", b"\xff\xd8camera-a-3\xff\xd9", frame_index=3)

    publisher = threading.Thread(target=publish_updates)
    try:
        with urllib.request.urlopen(f"{base}/stream/camera-a", timeout=2) as response:
            publisher.start()
            body = bytearray()
            deadline = time.monotonic() + 2.0
            while b"camera-a-3" not in body and time.monotonic() < deadline:
                body.extend(response.read(1))
            assert response.status == 200
            assert body.count(b"Content-Type: image/jpeg") >= 3
            assert body.count(b"--frame\r\n") >= 3
            assert b"camera-a-1" in body
            assert b"camera-a-2" in body
            assert b"camera-a-3" in body
            assert b"camera-b" not in body
    finally:
        publisher.join(timeout=1)
        server.stop()

    latest = buffer.get_latest("camera-a")
    assert latest is not None
    assert latest.jpeg == b"\xff\xd8camera-a-3\xff\xd9"
