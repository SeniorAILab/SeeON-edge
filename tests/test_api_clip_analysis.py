from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from shared.events.clip_analysis_wire import (
    ClipAnalysisBedGeometry,
    ClipAnalysisBox,
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
    decode_clip_analysis,
    encode_clip_analysis,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

CLIP_ID = "clip-1"
CLIP_SHA256 = "a" * 64
FIELD_1002E76A_ANALYSIS_NAME = "clip.analysis.94606b6630d57bd3.json"
FIELD_1002E76A_ANALYSIS_BYTES = (
    b'{"analysis_profile_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddd'
    b'dddddddddddd","bed_geometries":[{"points":[[100.0,100.0],[500.0,100.0],[500.0,30'
    b'0.0]],"provenance_pts":0}],"bed_model_sha256":"ccccccccccccccccccccccccccccccccc'
    b'ccccccccccccccccccccccccccccccc","clip_id":"clip-1","clip_sha256":"aaaaaaaaaaaaa'
    b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","decoder_identity":"ffmpeg"'
    b',"frames":[{"boxes":[{"confidence":0.9,"x1":1.0,"x2":10.0,"y1":1.0,"y2":10.0},{"'
    b'confidence":0.75,"x1":20.5,"x2":200.0,"y1":30.25,"y2":300.0}],"pts":0,"status":"'
    b'available"},{"boxes":[],"pts":400,"status":"no_evidence"}],"image_height":480,"i'
    b'mage_width":640,"pose_model_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    b'bbbbbbbbbbbbbbbbbbbbb","source":"clip_reanalysis","time_base":{"denominator":120'
    b'00,"numerator":1}}'
)
FIELD_1002E76A_ANALYSIS_SIDECAR = (
    b"3cfc00800bb20be0f6617cfd22a31af9ab1801d9f6dd80e5c5b3029ca80de761\n"
)


class _WorkerServer(ThreadingHTTPServer):
    response_status: int = 202
    response_body: dict[str, object] = {"state": "running"}
    get_response_status: int | None = None
    get_response_body: dict[str, object] | None = None
    requests: list[tuple[str, str, bytes, str | None]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _WorkerHandler)
        self.requests = []

    @property
    def origin(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _WorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._respond(
            self.server.response_status
            if self.server.get_response_status is None
            else self.server.get_response_status,
            self.server.response_body
            if self.server.get_response_body is None
            else self.server.get_response_body,
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.server.requests.append(
            (
                self.command,
                self.path,
                self.rfile.read(length),
                self.headers.get("X-Edge-Relay-Token"),
            )
        )
        self._respond(self.server.response_status, self.server.response_body)

    def _respond(self, response_status: int, response_body: dict[str, object]) -> None:
        if response_status == 204:
            self.send_response(response_status)
            self.end_headers()
            return
        body = json.dumps(response_body).encode()
        self.send_response(response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def worker_server() -> Iterator[_WorkerServer]:
    server = _WorkerServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture(autouse=True)
def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    clip_store = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(clip_store))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("ML_API_WORKER_STREAM_TIMEOUT_S", "0.1")
    get_settings.cache_clear()
    yield clip_store
    get_settings.cache_clear()


@pytest.fixture
def app(
    _environment: Path,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def _login(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        ).status_code
        == 204
    )


def _write_clip(clip_store: Path) -> Path:
    clip_dir = clip_store / "clips" / CLIP_ID
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"original")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": CLIP_ID,
                "camera_id": "camera-1",
                "event_ref": "event-1",
                "started_at": "2026-09-01T00:00:00Z",
                "duration_s": 1.0,
                "codec": "hevc",
                "path": f"clips/{CLIP_ID}",
                "finalized": True,
                "sha256": CLIP_SHA256,
            }
        ),
        encoding="utf-8",
    )
    return clip_dir


def _write_analysis(
    clip_dir: Path,
    *,
    clip_sha256: str = CLIP_SHA256,
    artifact_id: str = "0123456789abcdef",
) -> None:
    result = ClipAnalysisResult(
        source="clip_reanalysis",
        clip_id=CLIP_ID,
        clip_sha256=clip_sha256,
        pose_model_sha256="b" * 64,
        bed_model_sha256="c" * 64,
        decoder_identity="ffmpeg",
        analysis_profile_sha256="d" * 64,
        image_width=640,
        image_height=480,
        time_base=ClipAnalysisTimeBase(1, 12000),
        frames=(
            ClipAnalysisFrame(
                pts=0,
                status="available",
                boxes=(ClipAnalysisBox(1, 1, 10, 10, 0.9),),
            ),
        ),
        bed_geometries=(
            ClipAnalysisBedGeometry(((100.0, 100.0), (500.0, 100.0), (500.0, 300.0)), 0),
        ),
    )
    payload = encode_clip_analysis(result)
    artifact = clip_dir / f"clip.analysis.{artifact_id}.json"
    artifact.write_bytes(payload)
    artifact.with_name(f"{artifact.name}.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )


def _write_playback(
    clip_dir: Path,
    *,
    pts_identical: bool,
    source_sha256: str = CLIP_SHA256,
    rendition_sha256: str | None = None,
) -> str:
    digest = hashlib.sha256(b"playback").hexdigest()
    playback = clip_dir / f"clip.playback-h264.{digest[:16]}.mp4"
    playback.write_bytes(b"playback")
    (clip_dir / "clip.playback-h264.json").write_text(
        json.dumps(
            {
                "rendition": playback.name,
                "source_sha256": source_sha256,
                "rendition_sha256": digest if rendition_sha256 is None else rendition_sha256,
                "pts_identical": pts_identical,
                "time_base": "1/1000",
                "frames": 1,
                "source_frames": 1,
            }
        ),
        encoding="utf-8",
    )
    return digest


@pytest.mark.parametrize("worker_status", [200, 202, 409, 404])
def test_trigger_relays_worker_status_and_original_manifest_identity(
    _environment: Path,
    app: FastAPI,
    worker_server: _WorkerServer,
    monkeypatch: pytest.MonkeyPatch,
    worker_status: int,
) -> None:
    _write_clip(_environment)
    worker_server.response_status = worker_status
    worker_server.get_response_status = 200
    worker_server.get_response_body = {"state": "running"}
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        response = client.post(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == worker_status
    if worker_status in {200, 202, 409}:
        assert response.json() == {
            "state": "running",
            "served_media_sha256": CLIP_SHA256,
        }
    assert worker_server.requests == [
        (
            "POST",
            f"/clips/{CLIP_ID}/analysis",
            b'{"clip_sha256":"' + CLIP_SHA256.encode() + b'"}',
            "relay-token",
        )
    ]


def test_trigger_projects_worker_rejection_reason(
    _environment: Path, app: FastAPI, worker_server: _WorkerServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    worker_server.response_status = 422
    worker_server.response_body = {"error": "rejected", "reason": "duration"}
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        response = client.post(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 422
    assert response.json() == {
        "state": "failed",
        "served_media_sha256": CLIP_SHA256,
        "reason": "duration",
    }


def test_available_analysis_reports_identical_served_timing(
    _environment: Path, app: FastAPI
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(clip_dir, pts_identical=True)
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "available",
        "served_media_sha256": hashlib.sha256(b"playback").hexdigest(),
        "served_timing_identical": True,
        "result": {
            "source": "clip_reanalysis",
            "clip_id": CLIP_ID,
            "clip_sha256": CLIP_SHA256,
            "pose_model_sha256": "b" * 64,
            "bed_model_sha256": "c" * 64,
            "decoder_identity": "ffmpeg",
            "analysis_profile_sha256": "d" * 64,
            "image_width": 640,
            "image_height": 480,
            "time_base": {"numerator": 1, "denominator": 12000},
            "frames": [
                {
                    "pts": 0,
                    "status": "available",
                    "boxes": [{"x1": 1, "y1": 1, "x2": 10, "y2": 10, "confidence": 0.9}],
                }
            ],
            "bed_geometries": [
                {"points": [[100.0, 100.0], [500.0, 100.0], [500.0, 300.0]], "provenance_pts": 0}
            ],
        },
    }


def _field_result() -> ClipAnalysisResult:
    return ClipAnalysisResult(
        source="clip_reanalysis",
        clip_id=CLIP_ID,
        clip_sha256=CLIP_SHA256,
        pose_model_sha256="b" * 64,
        bed_model_sha256="c" * 64,
        decoder_identity="ffmpeg",
        analysis_profile_sha256="d" * 64,
        image_width=640,
        image_height=480,
        time_base=ClipAnalysisTimeBase(1, 12000),
        frames=(
            ClipAnalysisFrame(
                pts=0,
                status="available",
                boxes=(
                    ClipAnalysisBox(1.0, 1.0, 10.0, 10.0, 0.9),
                    ClipAnalysisBox(20.5, 30.25, 200.0, 300.0, 0.75),
                ),
            ),
            ClipAnalysisFrame(pts=400, status="no_evidence"),
        ),
        bed_geometries=(
            ClipAnalysisBedGeometry(((100.0, 100.0), (500.0, 100.0), (500.0, 300.0)), 0),
        ),
    )


def test_field_1002e76a_analysis_bytes_decode_and_are_served_as_written(
    _environment: Path, app: FastAPI
) -> None:
    digest = hashlib.sha256(FIELD_1002E76A_ANALYSIS_BYTES).hexdigest()
    assert f"{digest}\n".encode("ascii") == FIELD_1002E76A_ANALYSIS_SIDECAR
    decoded = decode_clip_analysis(FIELD_1002E76A_ANALYSIS_BYTES)
    assert decoded == _field_result()
    assert encode_clip_analysis(decoded) == FIELD_1002E76A_ANALYSIS_BYTES, (
        "encoder output no longer matches the artifact field revision 1002e76a wrote. "
        "If unintended, fix the encoder. If intended, decide on a migration or versioning: "
        "existing field artifacts break (the worker raises identity_collision when "
        "republishing one); do not replace FIELD_1002E76A_ANALYSIS_BYTES"
    )
    clip_dir = _write_clip(_environment)
    (clip_dir / FIELD_1002E76A_ANALYSIS_NAME).write_bytes(FIELD_1002E76A_ANALYSIS_BYTES)
    (clip_dir / f"{FIELD_1002E76A_ANALYSIS_NAME}.sha256").write_bytes(
        FIELD_1002E76A_ANALYSIS_SIDECAR
    )
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "available",
        "served_media_sha256": CLIP_SHA256,
        "served_timing_identical": True,
        "result": {
            "source": "clip_reanalysis",
            "clip_id": CLIP_ID,
            "clip_sha256": CLIP_SHA256,
            "pose_model_sha256": "b" * 64,
            "bed_model_sha256": "c" * 64,
            "decoder_identity": "ffmpeg",
            "analysis_profile_sha256": "d" * 64,
            "image_width": 640,
            "image_height": 480,
            "time_base": {"numerator": 1, "denominator": 12000},
            "frames": [
                {
                    "pts": 0,
                    "status": "available",
                    "boxes": [
                        {"x1": 1.0, "y1": 1.0, "x2": 10.0, "y2": 10.0, "confidence": 0.9},
                        {"x1": 20.5, "y1": 30.25, "x2": 200.0, "y2": 300.0, "confidence": 0.75},
                    ],
                },
                {"pts": 400, "status": "no_evidence", "boxes": []},
            ],
            "bed_geometries": [
                {"points": [[100.0, 100.0], [500.0, 100.0], [500.0, 300.0]], "provenance_pts": 0}
            ],
        },
    }


def test_available_analysis_on_original_reports_identical_served_timing(
    _environment: Path,
    app: FastAPI,
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "available"
    assert response.json()["served_timing_identical"] is True
    assert response.json()["result"]["clip_sha256"] == CLIP_SHA256


def test_available_analysis_reports_nonidentical_served_timing(
    _environment: Path, app: FastAPI
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(clip_dir, pts_identical=False)
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "unavailable",
        "served_media_sha256": hashlib.sha256(b"playback").hexdigest(),
        "reason": "timing_unverified",
        "served_timing_identical": False,
    }


def test_analysis_identity_mismatch_is_unavailable(_environment: Path, app: FastAPI) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir, clip_sha256="e" * 64)
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "unavailable",
        "served_media_sha256": CLIP_SHA256,
        "reason": "identity_mismatch",
    }


def test_corrupt_newest_analysis_does_not_mask_older_valid_analysis(
    _environment: Path,
    app: FastAPI,
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir, artifact_id="0123456789abcdef")
    newest = clip_dir / "clip.analysis.fedcba9876543210.json"
    newest.write_bytes(b"corrupt")
    newest.with_name(f"{newest.name}.sha256").write_text(
        hashlib.sha256(b"different").hexdigest() + "\n", encoding="ascii"
    )
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "available"
    assert response.json()["result"]["clip_sha256"] == CLIP_SHA256


@pytest.mark.parametrize(
    ("payload", "recorded_sha256"),
    [
        (b"corrupt", hashlib.sha256(b"different").hexdigest()),
        (b"corrupt", hashlib.sha256(b"corrupt").hexdigest()),
    ],
    ids=["sidecar-mismatch", "undecodable"],
)
def test_invalid_analysis_artifact_is_reported_as_artifact_invalid(
    _environment: Path, app: FastAPI, payload: bytes, recorded_sha256: str
) -> None:
    clip_dir = _write_clip(_environment)
    artifact = clip_dir / "clip.analysis.0123456789abcdef.json"
    artifact.write_bytes(payload)
    artifact.with_name(f"{artifact.name}.sha256").write_text(
        recorded_sha256 + "\n", encoding="ascii"
    )
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "unavailable",
        "served_media_sha256": CLIP_SHA256,
        "reason": "artifact_invalid",
    }


def test_worker_unreachable_is_an_honest_available_status(
    _environment: Path, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", "http://127.0.0.1:1")
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "unavailable"
    assert response.json()["reason"] == "worker_unreachable"


def test_worker_analysis_disabled_is_an_honest_unavailable_status(
    _environment: Path, app: FastAPI, worker_server: _WorkerServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    worker_server.response_status = 503
    worker_server.response_body = {"error": "clip_analysis_disabled"}
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "unavailable"
    assert response.json()["reason"] == "analysis_disabled"


@pytest.mark.parametrize(
    ("source_sha256", "rendition_sha256"),
    [("e" * 64, None), (CLIP_SHA256, "e" * 64)],
)
def test_analysis_uses_original_when_rendition_attestation_is_unbound(
    _environment: Path,
    app: FastAPI,
    source_sha256: str,
    rendition_sha256: str | None,
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(
        clip_dir,
        pts_identical=True,
        source_sha256=source_sha256,
        rendition_sha256=rendition_sha256,
    )
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.json()["state"] == "available"
    assert response.json()["served_timing_identical"] is True
    assert response.json()["result"]["clip_sha256"] == CLIP_SHA256


@pytest.mark.parametrize(
    "worker_body",
    [
        {"state": "idle"},
        {"state": "queued"},
        {"state": "running"},
        {"state": "failed"},
        {"state": "failed", "reason": "decode_error"},
        {"state": "queued", "reason": "busy"},
    ],
    ids=["idle", "queued", "running", "failed", "failed-with-reason", "queued-with-reason"],
)
def test_worker_states_include_served_media_identity(
    _environment: Path,
    app: FastAPI,
    worker_server: _WorkerServer,
    monkeypatch: pytest.MonkeyPatch,
    worker_body: dict[str, object],
) -> None:
    _write_clip(_environment)
    worker_server.response_status = 200
    worker_server.response_body = worker_body
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {**worker_body, "served_media_sha256": CLIP_SHA256}


def test_cancel_relays_empty_response_and_worker_auth_is_unreachable(
    _environment: Path, app: FastAPI, worker_server: _WorkerServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(app) as client:
        _login(client)
        worker_server.response_status = 204
        worker_server.response_body = {}
        cancelled = client.post(f"/api/v1/clips/{CLIP_ID}/analysis/cancel")
        worker_server.response_status = 403
        unreachable = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert cancelled.status_code == 204
    assert cancelled.content == b""
    assert unreachable.status_code == 200
    assert unreachable.json()["reason"] == "worker_unreachable"


def test_analysis_rejects_clip_id_outside_worker_grammar(_environment: Path, app: FastAPI) -> None:
    _write_clip(_environment)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip%3Alegacy/analysis")
    assert response.status_code == 400
