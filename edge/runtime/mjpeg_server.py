from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit

from contracts.frame import Frame
from contracts.observation import FrameObservation
from edge.domains.bed_exit.schema import DomainDebugSnapshot
from edge.runtime.overlay_renderer import OverlayRenderer
from edge.sources.probe import RTSPProbeError, probe_first_frame

BOUNDARY = b"frame"
DEV_MJPEG_ENV = "ML_WORKER_DEV_MJPEG"
DEV_MJPEG_HOST_ENV = "ML_WORKER_DEV_MJPEG_HOST"
DEV_MJPEG_PORT_ENV = "ML_WORKER_DEV_MJPEG_PORT"


@dataclass(frozen=True, slots=True)
class OverlayFrame:
    jpeg: bytes
    frame_index: int
    content_type: str = "image/jpeg"


class OverlayFrameBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frames: dict[str, OverlayFrame] = {}
        self._known_camera_ids: set[str] = set()

    def register_camera(self, camera_id: str) -> None:
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._condition.notify_all()

    def publish_jpeg(self, camera_id: str, jpeg: bytes, *, frame_index: int) -> None:
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._frames[camera_id] = OverlayFrame(bytes(jpeg), int(frame_index))
            self._condition.notify_all()

    def get_latest(self, camera_id: str) -> OverlayFrame | None:
        with self._condition:
            return self._frames.get(camera_id)

    def is_known(self, camera_id: str) -> bool:
        with self._condition:
            return camera_id in self._known_camera_ids

    def wait_for_latest(
        self, camera_id: str, *, previous: OverlayFrame | None, timeout: float
    ) -> OverlayFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frames.get(camera_id) is not previous,
                timeout=timeout,
            )
            return self._frames.get(camera_id)


class OverlayPublisher:
    def __init__(
        self,
        buffer: OverlayFrameBuffer,
        renderer: OverlayRenderer | None = None,
    ) -> None:
        self._buffer = buffer
        self._renderer = renderer if renderer is not None else OverlayRenderer()

    def publish(
        self,
        camera_id: str,
        frame: Frame,
        observation: FrameObservation,
        debug_snapshots: tuple[DomainDebugSnapshot, ...],
    ) -> None:
        jpeg = self._renderer.encode_jpeg(frame, observation, debug_snapshots)
        self._buffer.publish_jpeg(camera_id, jpeg, frame_index=frame.index)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegServer:
    def __init__(
        self,
        buffer: OverlayFrameBuffer,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        probe_token: str | None = None,
        probe: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self.buffer = buffer
        self.host = host
        self.port = int(port)
        self.probe_token = probe_token
        self.probe = probe if probe is not None else _probe_rtsp_first_frame
        self._server = _ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = int(self._server.server_port)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        buffer = self.buffer
        probe_token = self.probe_token
        probe = self.probe
        poll_interval_seconds = 0.05
        heartbeat_interval_seconds = 1.0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib hook
                path = urlsplit(self.path).path
                stream_prefix = "/stream/"
                snapshot_prefix = "/snapshot/"
                if path.startswith(stream_prefix):
                    self._handle_stream(unquote(path[len(stream_prefix) :]))
                elif path.startswith(snapshot_prefix):
                    self._handle_snapshot(unquote(path[len(snapshot_prefix) :]))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def _resolve_frame(self, camera_id: str) -> OverlayFrame | None:
                if camera_id == "" or not buffer.is_known(camera_id):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return None
                frame = buffer.get_latest(camera_id)
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                    return None
                return frame

            def _handle_stream(self, camera_id: str) -> None:
                frame = self._resolve_frame(camera_id)
                if frame is None:
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                last_sent: OverlayFrame | None = None
                last_send_at = 0.0
                while True:
                    frame = buffer.wait_for_latest(
                        camera_id,
                        previous=last_sent,
                        timeout=poll_interval_seconds,
                    )
                    now = time.monotonic()
                    should_heartbeat = now - last_send_at >= heartbeat_interval_seconds
                    if frame is None or (frame is last_sent and not should_heartbeat):
                        continue
                    try:
                        self._write_part(frame)
                        self.wfile.flush()
                    except (TimeoutError, BrokenPipeError, ConnectionResetError):
                        return
                    last_sent = frame
                    last_send_at = now

            def _handle_snapshot(self, camera_id: str) -> None:
                frame = self._resolve_frame(camera_id)
                if frame is None:
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", frame.content_type)
                self.send_header("Content-Length", str(len(frame.jpeg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(frame.jpeg)
                except (TimeoutError, BrokenPipeError, ConnectionResetError):
                    return
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook
                path = urlsplit(self.path).path
                if path != "/probe":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not _authorized_probe(self.headers.get("X-Edge-Relay-Token"), probe_token):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                payload = self._read_json()
                if payload is None:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                rtsp_url = payload.get("rtsp_url")
                if not isinstance(rtsp_url, str) or rtsp_url.strip() == "":
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                result: dict[str, object]
                try:
                    result = probe(rtsp_url)
                except RTSPProbeError as exc:
                    result = {"ok": False, "error_class": exc.error_class}
                except Exception:  # noqa: BLE001 - probes normalize unexpected decoder errors.
                    result = {"ok": False, "error_class": "decode"}
                self._write_json(result)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
                return

            def _read_json(self) -> dict[str, object] | None:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    return None
                try:
                    length = int(raw_length)
                except ValueError:
                    return None
                if length <= 0 or length > 8192:
                    return None
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
                return payload if isinstance(payload, dict) else None

            def _write_json(self, payload: dict[str, object]) -> None:
                body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_part(self, frame: OverlayFrame) -> None:
                self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame.jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame.jpeg)
                self.wfile.write(b"\r\n")

        return Handler


def _authorized_probe(supplied: str | None, expected: str | None) -> bool:
    if expected is None or expected.strip() == "" or supplied is None:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.strip().encode("utf-8"))


def _probe_rtsp_first_frame(rtsp_url: str) -> dict[str, object]:
    result = probe_first_frame(rtsp_url)
    return {
        "ok": True,
        "backend": result.backend,
        "width": result.width,
        "height": result.height,
    }


def dev_mjpeg_enabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(DEV_MJPEG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def dev_mjpeg_host(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(DEV_MJPEG_HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"


def dev_mjpeg_port(environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get(DEV_MJPEG_PORT_ENV, "8090").strip() or "8090"
    return int(raw)
