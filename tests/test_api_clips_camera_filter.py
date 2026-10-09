from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips.service.camera_filter import camera_filter_ids
from backend.app.features.clips.store import ClipStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


class _Registry:
    def __init__(self, cameras: list[dict[str, object]]) -> None:
        self.cameras = cameras

    def snapshot(self) -> dict[str, object]:
        return {"registry_version": 1, "cameras": self.cameras}


def test_camera_filter_ids_resolves_local_and_hub_ids() -> None:
    registry = _Registry(
        [
            {"id": "cam-plain", "backend_camera_id": None},
            {"id": "cam-local", "backend_camera_id": "hub-7"},
        ]
    )
    assert camera_filter_ids(registry, "cam-local") == ("cam-local", "hub-7")
    assert camera_filter_ids(registry, "hub-7") == ("cam-local", "hub-7")
    assert camera_filter_ids(registry, "cam-plain") == ("cam-plain",)
    assert camera_filter_ids(registry, "unknown") == ("unknown",)
    assert camera_filter_ids(None, "cam-local") == ("cam-local",)
    assert camera_filter_ids(registry, None) is None


def _write_clip(root: Path, clip_id: str, camera_id: str, second: int) -> None:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    payload = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": f"event-{clip_id}",
        "event_type": "fall",
        "started_at": f"2026-08-09T00:00:{second:02d}Z",
        "duration_s": 0.0,
        "codec": "",
        "video_available": False,
        "finalized": True,
    }
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_camera_filter_returns_clips_under_local_and_hub_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    root = tmp_path / "clip-store"
    _write_clip(root, "clip-before-mapping", "cam-local", 1)
    _write_clip(root, "clip-after-mapping", "hub-7", 2)
    _write_clip(root, "clip-other-camera", "cam-other", 3)
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.camera_registry.create(
        camera_id="cam-local",
        label="Mapped camera",
        rtsp_url="rtsp://camera.invalid/live",
        space_id=None,
        status="unknown",
        backend_camera_id="hub-7",
    )
    app.state.clip_store = ClipStore(root)
    index_clips(app)
    expected = {"clip-before-mapping", "clip-after-mapping"}
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        for requested in ("cam-local", "hub-7"):
            response = client.get("/api/v1/clips", params={"camera_id": requested})
            assert response.status_code == 200
            body = response.json()
            assert {clip["clip_id"] for clip in body["clips"]} == expected
            assert body["pagination"]["total"] == 2
