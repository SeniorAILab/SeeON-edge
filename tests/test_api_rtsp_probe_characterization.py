from __future__ import annotations

import json
from typing import Any, Final

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

RELAY_HEADER_NAME: Final = "X-Edge-Relay-Token"
RELAY_TOKEN: Final = "relay-token"
RELAY_HEADERS = {RELAY_HEADER_NAME: RELAY_TOKEN}

class _Endpoint:
    def __init__(self, original_url: str) -> None:
        self.original_url = original_url


def _app() -> Any:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    return app


def test_camera_probe_unavailable_when_origin_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()

    # Fake store with one camera
    class _FakeStore:
        def get(self, camera_id: str) -> dict[str, Any] | None:
            return (
                {"id": camera_id, "rtsp_url": "rtsp://camera.invalid/a"}
                if camera_id == "cam1"
                else None
            )

    import backend.app.features.cameras.router as cameras_router

    # Bypass dashboard auth
    monkeypatch.setattr(cameras_router, "_authorize", lambda request: "user", raising=True)
    monkeypatch.setattr(cameras_router, "_store", lambda app: _FakeStore(), raising=True)
    # Allow any RTSP URL at validation
    monkeypatch.setattr(
        cameras_router, "assert_rtsp_endpoint_allowed", lambda url: _Endpoint(url), raising=True
    )

    # Settings with blank origin triggers probe_unavailable=True
    class _S:
        worker_probe_origin: str = ""
        worker_probe_timeout_s: int = 5

    from backend.app.core import config as core_config

    monkeypatch.setattr(core_config, "get_settings", lambda: _S(), raising=True)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/cam1/test",
            headers=RELAY_HEADERS,
            json={"rtsp_url": "rtsp://override.invalid/a"},
        )
    assert resp.status_code == 200
    assert resp.content == json.dumps(
        {"ok": False, "probe_unavailable": True}, separators=(",", ":")
    ).encode("utf-8")


def test_camera_probe_maps_worker_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()

    # Fake store with one camera
    class _FakeStore:
        def get(self, camera_id: str) -> dict[str, Any] | None:
            return (
                {"id": camera_id, "rtsp_url": "rtsp://camera.invalid/b"}
                if camera_id == "cam2"
                else None
            )

    import backend.app.features.cameras.router as cameras_router

    # Bypass dashboard auth
    monkeypatch.setattr(cameras_router, "_authorize", lambda request: "user", raising=True)
    monkeypatch.setattr(cameras_router, "_store", lambda app: _FakeStore(), raising=True)

    # Settings with non-empty origin
    class _S:
        worker_probe_origin: str = "http://worker.invalid"
        worker_probe_timeout_s: int = 5

    from backend.app.core import config as core_config

    monkeypatch.setattr(core_config, "get_settings", lambda: _S(), raising=True)

    # Fake successful HTTP call returning a worker payload
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"ok": False, "error_class": "timeout", "width": 640, "height": 480}
            ).encode("utf-8")

    import urllib.request as urllib_request

    monkeypatch.setattr(urllib_request, "urlopen", lambda *_, **__: _Resp(), raising=True)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/cam2/test",
            headers=RELAY_HEADERS,
            json={"rtsp_url": "rtsp://override.invalid/b"},
        )
    assert resp.status_code == 200
    assert resp.content == json.dumps(
        {"ok": False, "error_class": "timeout", "width": 640, "height": 480}, separators=(",", ":")
    ).encode("utf-8")


def test_create_camera_invalid_rtsp_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()
    import backend.app.features.cameras.router as cameras_router

    # Bypass auth to hit validation quickly
    monkeypatch.setattr(cameras_router, "_authorize", lambda request: "user", raising=True)
    # Force validation failure before DB touch
    monkeypatch.setattr(
        cameras_router,
        "assert_rtsp_endpoint_allowed",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
        raising=True,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras",
            json={
                "label": "X",
                "rtsp_url": "rtsp://bad.invalid",
                "space_id": None,
                "force_register": False,
            },
            headers=RELAY_HEADERS,
        )
    assert resp.status_code == 400


def test_update_camera_invalid_rtsp_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()
    import backend.app.features.cameras.router as cameras_router

    # Bypass auth and provide a fake current record
    monkeypatch.setattr(cameras_router, "_authorize", lambda request: "user", raising=True)

    class _FakeStore:
        def get(self, camera_id: str) -> dict[str, Any] | None:
            return (
                {"id": camera_id, "rtsp_url": "rtsp://camera.invalid/x"}
                if camera_id == "cam3"
                else None
            )

    monkeypatch.setattr(cameras_router, "_store", lambda app: _FakeStore(), raising=True)
    monkeypatch.setattr(
        cameras_router,
        "assert_rtsp_endpoint_allowed",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
        raising=True,
    )

    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/cameras/cam3",
            json={"rtsp_url": "rtsp://bad.invalid"},
            headers=RELAY_HEADERS,
        )
    assert resp.status_code == 400

