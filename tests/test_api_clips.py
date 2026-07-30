from __future__ import annotations

import json
from typing import Self

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

AUTH = {"Authorization": "Bearer relay-token"}


class FakeHTTPResponse:
    status = 202

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _write_manifest(
    clip_store,
    clip_id: str,
    *,
    camera_id: str = "camera-1",
    event_ref: str | None = None,
    event_type: str | None = "fall",
    started_at: str = "2026-07-06T00:00:00Z",
    path: str | None = None,
    finalized: bool = True,
) -> None:
    clip_dir = clip_store / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(f"video:{clip_id}".encode())
    payload = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": event_ref or f"event-{clip_id}",
        "started_at": started_at,
        "duration_s": 30.0,
        "codec": "h264",
        "path": path or f"clips/{clip_id}",
        "finalized": finalized,
    }
    if event_type is not None:
        payload["event_type"] = event_type
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def clip_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)
    monkeypatch.delenv("API_BACKEND_FACILITY_TOKEN", raising=False)
    monkeypatch.delenv("API_FACILITY_TOKEN", raising=False)
    return tmp_path


def test_list_clips_returns_only_finalized_latest_first_and_filters_camera(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(
        clip_store,
        "clip-old",
        camera_id="camera-1",
        started_at="2026-07-06T00:00:00Z",
    )
    _write_manifest(
        clip_store,
        "clip-new",
        camera_id="camera-2",
        started_at="2026-07-06T00:01:00Z",
    )
    _write_manifest(
        clip_store,
        "clip-open",
        camera_id="camera-1",
        started_at="2026-07-06T00:02:00Z",
        finalized=False,
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        listed = client.get("/api/v1/clips", headers=AUTH)
        filtered = client.get("/api/v1/clips", params={"camera_id": "camera-1"}, headers=AUTH)

    assert listed.status_code == 200
    assert [clip["clip_id"] for clip in listed.json()["clips"]] == ["clip-new", "clip-old"]
    assert listed.json()["clips"][0]["event_type"] == "fall"
    assert filtered.status_code == 200
    assert [clip["clip_id"] for clip in filtered.json()["clips"]] == ["clip-old"]


def test_list_clips_preserves_event_type_when_event_ref_is_identity(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(
        clip_store,
        "clip-bed-exit",
        event_ref="0:0",
        event_type="bed-exit",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get("/api/v1/clips", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["clips"][0]["event_ref"] == "0:0"
    assert response.json()["clips"][0]["event_type"] == "bed-exit"


def test_streams_manifest_video_and_appends_audit(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        video = client.get("/api/v1/clips/clip-1/video", headers=AUTH)
        query_video = client.get("/api/v1/clips/clip-1/video", params={"token": "relay-token"})
        audit = client.get("/api/v1/audit", headers=AUTH)

    assert video.status_code == 200
    assert video.content == b"video:clip-1"
    assert video.headers["content-type"].startswith("video/mp4")
    assert query_video.status_code == 200
    assert query_video.content == b"video:clip-1"
    assert audit.status_code == 200
    assert audit.json()["entries"] == [
        {
            "ts": audit.json()["entries"][0]["ts"],
            "actor": "bearer",
            "action": "play",
            "clip_id": "clip-1",
        },
        {
            "ts": audit.json()["entries"][1]["ts"],
            "actor": "operator",
            "action": "play",
            "clip_id": "clip-1",
        },
    ]


def test_label_clip_saves_sidecar_and_audit(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    label_store = clip_env / "label-store"
    _write_manifest(clip_store, "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.put(
            "/api/v1/clips/clip-1/label",
            headers=AUTH,
            json={"label": "TRUE_POSITIVE", "reviewer": "reviewer-1"},
        )
        clear_response = client.put(
            "/api/v1/clips/clip-1/label",
            headers=AUTH,
            json={"label": None, "reviewer": "reviewer-2"},
        )
        default_reviewer = client.put(
            "/api/v1/clips/clip-1/label",
            headers=AUTH,
            json={"label": "FALSE_POSITIVE"},
        )
        audit = client.get("/api/v1/audit", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["label"] == "TRUE_POSITIVE"
    assert clear_response.status_code == 200
    assert clear_response.json()["label"] is None
    assert default_reviewer.status_code == 200
    assert default_reviewer.json()["reviewer"] == "bearer"
    saved = json.loads((label_store / "labels" / "clip-1.json").read_text(encoding="utf-8"))
    assert saved["label"] == "FALSE_POSITIVE"
    assert saved["reviewer"] == "bearer"
    audit_rows = [
        (entry["actor"], entry["action"], entry["clip_id"]) for entry in audit.json()["entries"]
    ]
    assert audit_rows == [
        ("reviewer-1", "label", "clip-1"),
        ("reviewer-2", "label", "clip-1"),
        ("bearer", "label", "clip-1"),
    ]


def test_clip_routes_require_bearer_token(clip_env) -> None:
    _write_manifest(clip_env / "clip-store", "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        unauthenticated = client.get("/api/v1/clips")
        wrong = client.get("/api/v1/clips", headers={"Authorization": "Bearer wrong"})
        legacy_header = client.get("/api/v1/clips", headers={"X-Edge-Relay-Token": "relay-token"})

    assert unauthenticated.status_code == 401
    assert wrong.status_code == 403
    assert legacy_header.status_code == 401


def test_video_rejects_manifest_path_escape(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    (clip_env / "secret.mp4").write_bytes(b"secret")
    _write_manifest(clip_store, "clip-escape", path="../secret.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get("/api/v1/clips/clip-escape/video", headers=AUTH)
        invalid_id = client.get("/api/v1/clips/%2E%2E/video", headers=AUTH)

    assert response.status_code == 400
    assert invalid_id.status_code == 400


def test_label_and_audit_backend_backup_is_best_effort(
    clip_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(clip_env / "clip-store", "clip-1")
    monkeypatch.setenv("API_BACKEND_CLIP_EVENTS_URL", "http://backend/api/v1/clip-events")
    monkeypatch.setenv("API_BACKEND_FACILITY_TOKEN", "facility-token")
    calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "authorization": request.headers.get("Authorization"),
                "timeout": timeout,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.put(
            "/api/v1/clips/clip-1/label",
            headers=AUTH,
            json={"label": "FALSE_POSITIVE", "reviewer": "reviewer-1"},
        )

    assert response.status_code == 200
    assert [call["body"]["type"] for call in calls] == ["clip_label", "clip_audit"]
    assert {call["authorization"] for call in calls} == {"Bearer facility-token"}
    assert {call["url"] for call in calls} == {"http://backend/api/v1/clip-events"}
