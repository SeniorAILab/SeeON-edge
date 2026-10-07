from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.cameras.camera_values import ProbeResult
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

CAMERAS = "/api/v1/cameras"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
SESSION_REQUIRED = b'{"detail":"dashboard session required"}'
NOT_FOUND = b'{"detail":"camera not found"}'
OK_PROBE = ProbeResult(True, width=640, height=480)
FAILED_PROBE = ProbeResult(False, "timeout")


@dataclass
class Harness:
    app: FastAPI
    sandbox: ProductSandbox
    probe_results: list[ProbeResult] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)
    syncs: int = 0

    def audit_rows(self, action: AuditAction) -> list[tuple[str, str]]:
        rows = self.sandbox.admin.execute(
            "SELECT target_id, actor_id FROM audit_events WHERE action=%s ORDER BY audit_id",
            (action.value,),
        ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def seed(
        self, camera_id: str, rtsp_url: str, space_id: str | None = None, **extra: object
    ) -> None:
        self.app.state.camera_registry.create(
            camera_id=camera_id,
            label=f"seed {camera_id}",
            rtsp_url=rtsp_url,
            space_id=space_id,
            status="offline",
            **extra,
        )


@pytest.fixture
def harness(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Harness]:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    harness = Harness(app=app, sandbox=postgres_product_sandbox)

    def probe(_request: object, rtsp_url: str) -> ProbeResult:
        harness.probed.append(rtsp_url)
        return harness.probe_results.pop(0) if harness.probe_results else OK_PROBE

    def sync(_app: object, **_kwargs: object) -> None:
        harness.syncs += 1

    monkeypatch.setattr("backend.app.features.cameras.router._probe_rtsp_url", probe)
    monkeypatch.setattr("backend.app.features.cameras.router.sync_camera_roster", sync)
    yield harness


@pytest.fixture
def client(harness: Harness) -> Iterator[TestClient]:
    with TestClient(harness.app) as client:
        login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
        assert login.status_code == 204
        yield client


def _registry(client: TestClient) -> dict[str, object]:
    response = client.get(CAMERAS)
    assert response.status_code == 200
    return response.json()


def _registry_version(client: TestClient) -> int:
    return int(_registry(client)["registry_version"])


def _listed_ids(client: TestClient) -> list[str]:
    cameras = _registry(client)["cameras"]
    assert isinstance(cameras, list)
    return sorted(str(camera["id"]) for camera in cameras)


def _encoded(body: object) -> bytes:
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _camera_body(
    *,
    camera_id: str,
    label: str,
    masked: str,
    status: str,
    created_at: str,
    never_connected: bool,
    last_ok_at: str | None,
    last_probed_at: str | None,
    space_id: str | None = None,
    backend_camera_id: str | None = None,
    decode_backend: str | None = None,
    floor: int | None = None,
    refs: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "id": camera_id,
        "label": label,
        "rtsp_url_masked": masked,
        "space_id": space_id,
        "backend_camera_id": backend_camera_id,
        "mapping_pending": False,
        "mapping_state": "unmapped",
        "status": status,
        "decode_backend": decode_backend,
        "created_at": created_at,
        "space_name": None,
        "floor_name": None,
        "floor": floor,
        "sync": None,
        "last_heartbeat_at": None,
        "heartbeat_age_sec": None,
        "never_connected": never_connected,
        "last_ok_at": last_ok_at,
        "last_probed_at": last_probed_at,
        "bed_zone": None,
        **(refs or {}),
    }


def test_create_with_reachable_stream_returns_exact_online_body(
    harness: Harness, client: TestClient
) -> None:
    response = client.post(
        CAMERAS,
        json={
            "label": "Bed 1",
            "rtsp_url": "rtsp://camera.example:8554/live",
            "space_id": "space-1",
            "decode_backend": " NVDEC ",
            "floor": -1,
            "force_register": True,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.match(body["id"])
    assert ISO_RE.match(body["created_at"])
    assert ISO_RE.match(body["last_probed_at"])
    assert response.content == _encoded(
        _camera_body(
            camera_id=body["id"],
            label="Bed 1",
            masked="rtsp://redacted-camera:8554/live",
            status="online",
            created_at=body["created_at"],
            never_connected=False,
            last_ok_at=body["last_probed_at"],
            last_probed_at=body["last_probed_at"],
            space_id="space-1",
            decode_backend="nvdec",
            floor=-1,
        )
    )
    assert harness.probed == ["rtsp://camera.example:8554/live"]
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == [(body["id"], "admin")]
    assert harness.syncs == 1
    assert _listed_ids(client) == [body["id"]]
    assert _registry_version(client) == 1


@pytest.mark.parametrize(
    "probe", [FAILED_PROBE, ProbeResult(False, probe_unavailable=True)], ids=["failed", "absent"]
)
def test_create_with_unreachable_stream_still_registers_offline(
    harness: Harness, client: TestClient, probe: ProbeResult
) -> None:
    harness.probe_results.append(probe)

    response = client.post(CAMERAS, json={"label": "Hall", "rtsp_url": "rtsp://camera.example/a"})

    assert response.status_code == 201
    body = response.json()
    assert ISO_RE.match(body["last_probed_at"])
    assert response.content == _encoded(
        _camera_body(
            camera_id=body["id"],
            label="Hall",
            masked="rtsp://redacted-camera/a",
            status="offline",
            created_at=body["created_at"],
            never_connected=True,
            last_ok_at=None,
            last_probed_at=body["last_probed_at"],
        )
    )
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == [(body["id"], "admin")]
    assert harness.syncs == 1


@pytest.mark.parametrize(
    ("payload", "status_code", "content"),
    [
        (
            {"label": "x", "rtsp_url": "rtsp://127.0.0.1/live"},
            400,
            b'{"detail":"loopback destination is not permitted"}',
        ),
        (
            {"label": "x", "rtsp_url": "rtsp://camera.example/a", "decode_backend": "gpu"},
            400,
            b'{"detail":"invalid decode_backend"}',
        ),
        (
            {"label": "x", "rtsp_url": "rtsp://camera.example/a", "floor": 11},
            400,
            b'{"detail":"invalid floor"}',
        ),
        (
            {"label": "", "rtsp_url": "rtsp://camera.example/a"},
            422,
            (
                b'{"detail":[{"type":"string_too_short","loc":["body","label"],'
                b'"msg":"String should have at least 1 character","input":"",'
                b'"ctx":{"min_length":1}}]}'
            ),
        ),
        (
            {"rtsp_url": "rtsp://camera.example/a"},
            422,
            (
                b'{"detail":[{"type":"missing","loc":["body","label"],"msg":"Field required",'
                b'"input":{"rtsp_url":"rtsp://camera.example/a"}}]}'
            ),
        ),
        (
            {"label": "x", "rtsp_url": "rtsp://camera.example/a", "bogus": 1},
            422,
            (
                b'{"detail":[{"type":"extra_forbidden","loc":["body","bogus"],'
                b'"msg":"Extra inputs are not permitted","input":1}]}'
            ),
        ),
    ],
    ids=["loopback", "decode-backend", "floor", "empty-label", "missing-label", "extra-field"],
)
def test_create_rejects_invalid_input_before_probe_or_write(
    harness: Harness,
    client: TestClient,
    payload: dict[str, object],
    status_code: int,
    content: bytes,
) -> None:
    response = client.post(CAMERAS, json=payload)

    assert (response.status_code, response.content) == (status_code, content)
    assert harness.probed == []
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == []
    assert harness.syncs == 0
    assert _registry_version(client) == 0


def test_create_duplicate_stream_probes_then_conflicts_without_write(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example:554/live")

    response = client.post(
        CAMERAS, json={"label": "Copy", "rtsp_url": "rtsp://CAMERA.example/live/"}
    )

    assert response.status_code == 409
    assert response.content == (
        b'{"detail":{"error":"duplicate_camera","existing_camera_id":"camera-a",'
        b'"existing_label":"seed camera-a"}}'
    )
    assert harness.probed == ["rtsp://CAMERA.example/live/"]
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == []
    assert harness.syncs == 0
    assert _listed_ids(client) == ["camera-a"]
    assert _registry_version(client) == 1


def test_create_unbound_room_ref_conflicts_on_provisional_id_without_write(
    harness: Harness, client: TestClient
) -> None:
    response = client.post(
        CAMERAS,
        json={"label": "Ref", "rtsp_url": "rtsp://camera.example/ref", "edge_ref": "camera-ref"},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert list(detail) == ["code", "edge_ref"]
    assert detail["code"] == "INVALID_BINDING"
    assert UUID_RE.match(detail["edge_ref"])
    assert harness.probed == ["rtsp://camera.example/ref"]
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == []
    assert harness.syncs == 0
    assert _listed_ids(client) == []
    assert _registry_version(client) == 0


def test_create_with_explicit_refs_echoes_them(harness: Harness, client: TestClient) -> None:
    store = harness.app.state.camera_registry
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")

    response = client.post(
        CAMERAS,
        json={
            "label": "Ref",
            "rtsp_url": "rtsp://camera.example/ref",
            "edge_ref": "camera-a",
            "room_edge_ref": "room-a",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert response.content == _encoded(
        _camera_body(
            camera_id=body["id"],
            label="Ref",
            masked="rtsp://redacted-camera/ref",
            status="online",
            created_at=body["created_at"],
            never_connected=False,
            last_ok_at=body["last_probed_at"],
            last_probed_at=body["last_probed_at"],
            refs={"edge_ref": "camera-a", "room_edge_ref": "room-a"},
        )
    )
    assert _registry_version(client) == 3


def test_create_audit_failure_returns_empty_503_and_rolls_back(
    harness: Harness, client: TestClient
) -> None:
    harness.sandbox.admin.execute(
        "CREATE FUNCTION reject_camera_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    harness.sandbox.admin.execute(
        "CREATE TRIGGER reject_camera_audit BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_camera_audit()"
    )

    response = client.post(CAMERAS, json={"label": "A", "rtsp_url": "rtsp://camera.example/a"})

    assert (response.status_code, response.content) == (503, b"")
    assert harness.probed == ["rtsp://camera.example/a"]
    assert harness.syncs == 0
    assert harness.app.state.camera_registry.snapshot()["cameras"] == []


def test_mutations_require_a_dashboard_session(harness: Harness) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")

    with TestClient(harness.app) as anonymous:
        created = anonymous.post(
            CAMERAS, json={"label": "A", "rtsp_url": "rtsp://camera.example/b"}
        )
        updated = anonymous.patch(f"{CAMERAS}/camera-a", json={"label": "B"})
        deleted = anonymous.delete(f"{CAMERAS}/camera-a")
        missing = anonymous.delete(f"{CAMERAS}/missing")

    for response in (created, updated, deleted, missing):
        assert (response.status_code, response.content) == (401, SESSION_REQUIRED)
    assert harness.probed == []
    assert harness.syncs == 0
    assert harness.app.state.camera_registry.get("camera-a")["label"] == "seed camera-a"


def test_update_unknown_camera_is_404_before_validation_or_probe(
    harness: Harness, client: TestClient
) -> None:
    response = client.patch(
        f"{CAMERAS}/missing", json={"rtsp_url": "rtsp://127.0.0.1/live", "floor": 99}
    )

    assert (response.status_code, response.content) == (404, NOT_FOUND)
    assert harness.probed == []
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == []
    assert harness.syncs == 0


def test_update_with_empty_body_returns_current_without_write(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a", decode_backend="cpu", floor=3)
    before = harness.app.state.camera_registry.get("camera-a")

    response = client.patch(f"{CAMERAS}/camera-a", json={})

    assert response.status_code == 200
    assert response.content == _encoded(
        _camera_body(
            camera_id="camera-a",
            label="seed camera-a",
            masked="rtsp://redacted-camera/a",
            status="offline",
            created_at=str(before["created_at"]),
            never_connected=True,
            last_ok_at=None,
            last_probed_at=None,
            decode_backend="cpu",
            floor=3,
        )
    )
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == []
    assert harness.syncs == 0
    assert _registry_version(client) == 1


def test_update_label_only_skips_probe_and_keeps_status(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")

    response = client.patch(f"{CAMERAS}/camera-a", json={"label": "Renamed"})

    assert response.status_code == 200
    body = response.json()
    assert response.content == _encoded(
        _camera_body(
            camera_id="camera-a",
            label="Renamed",
            masked="rtsp://redacted-camera/a",
            status="offline",
            created_at=body["created_at"],
            never_connected=True,
            last_ok_at=None,
            last_probed_at=None,
        )
    )
    assert harness.probed == []
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == [("camera-a", "admin")]
    assert harness.syncs == 1
    assert _registry_version(client) == 2


def test_update_rtsp_url_reprobes_and_records_success(harness: Harness, client: TestClient) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")

    response = client.patch(f"{CAMERAS}/camera-a", json={"rtsp_url": "rtsp://camera.example/b"})

    assert response.status_code == 200
    body = response.json()
    assert ISO_RE.match(body["last_probed_at"])
    assert response.content == _encoded(
        _camera_body(
            camera_id="camera-a",
            label="seed camera-a",
            masked="rtsp://redacted-camera/b",
            status="online",
            created_at=body["created_at"],
            never_connected=False,
            last_ok_at=body["last_probed_at"],
            last_probed_at=body["last_probed_at"],
        )
    )
    assert harness.probed == ["rtsp://camera.example/b"]
    assert harness.syncs == 1


def test_update_rtsp_url_with_failed_probe_keeps_connection_history(
    harness: Harness, client: TestClient
) -> None:
    harness.seed(
        "camera-a",
        "rtsp://camera.example/a",
        never_connected=False,
        last_ok_at="2026-01-01T00:00:00.000Z",
        last_probed_at="2026-01-01T00:00:00.000Z",
    )
    harness.probe_results.append(FAILED_PROBE)

    response = client.patch(f"{CAMERAS}/camera-a", json={"rtsp_url": "rtsp://camera.example/b"})

    assert response.status_code == 200
    body = response.json()
    assert body["last_probed_at"] != "2026-01-01T00:00:00.000Z"
    assert response.content == _encoded(
        _camera_body(
            camera_id="camera-a",
            label="seed camera-a",
            masked="rtsp://redacted-camera/b",
            status="offline",
            created_at=body["created_at"],
            never_connected=False,
            last_ok_at="2026-01-01T00:00:00.000Z",
            last_probed_at=body["last_probed_at"],
        )
    )
    assert harness.probed == ["rtsp://camera.example/b"]


def test_update_explicit_nulls_clear_optionals_and_still_write(
    harness: Harness, client: TestClient
) -> None:
    harness.seed(
        "camera-a", "rtsp://camera.example/a", space_id="space-1", decode_backend="cpu", floor=2
    )

    ignored = client.patch(f"{CAMERAS}/camera-a", json={"label": None, "rtsp_url": None})
    cleared = client.patch(
        f"{CAMERAS}/camera-a", json={"space_id": None, "decode_backend": None, "floor": None}
    )

    assert ignored.status_code == cleared.status_code == 200
    assert ignored.json()["label"] == "seed camera-a"
    assert ignored.json()["decode_backend"] == "cpu"
    assert ignored.json()["space_id"] == "space-1"
    assert (
        cleared.json()["space_id"],
        cleared.json()["decode_backend"],
        cleared.json()["floor"],
    ) == (None, None, None)
    assert harness.probed == []
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == [
        ("camera-a", "admin"),
        ("camera-a", "admin"),
    ]
    assert harness.syncs == 2
    assert _registry_version(client) == 3


@pytest.mark.parametrize(
    ("payload", "content", "probed"),
    [
        (
            {"rtsp_url": "rtsp://127.0.0.1/a"},
            b'{"detail":"loopback destination is not permitted"}',
            [],
        ),
        ({"decode_backend": "gpu"}, b'{"detail":"invalid decode_backend"}', []),
        ({"floor": -2}, b'{"detail":"invalid floor"}', []),
        (
            {"rtsp_url": "rtsp://camera.example/b", "floor": 11},
            b'{"detail":"invalid floor"}',
            ["rtsp://camera.example/b"],
        ),
    ],
    ids=["loopback", "decode-backend", "floor", "probe-then-floor"],
)
def test_update_rejects_invalid_fields_without_write(
    harness: Harness,
    client: TestClient,
    payload: dict[str, object],
    content: bytes,
    probed: list[str],
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")

    response = client.patch(f"{CAMERAS}/camera-a", json=payload)

    assert (response.status_code, response.content) == (400, content)
    assert harness.probed == probed
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == []
    assert harness.syncs == 0
    assert _registry_version(client) == 1


def test_update_to_duplicate_stream_conflicts_after_probe_without_write(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")
    harness.seed("camera-b", "rtsp://camera.example/b")

    response = client.patch(f"{CAMERAS}/camera-b", json={"rtsp_url": "rtsp://camera.example/a"})

    assert response.status_code == 409
    assert response.content == (
        b'{"detail":{"error":"duplicate_camera","existing_camera_id":"camera-a",'
        b'"existing_label":"seed camera-a"}}'
    )
    assert harness.probed == ["rtsp://camera.example/a"]
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == []
    assert harness.syncs == 0
    assert _registry_version(client) == 2


def test_update_invalid_binding_conflicts_on_camera_id_without_write(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")

    response = client.patch(f"{CAMERAS}/camera-a", json={"edge_ref": "camera-ref"})

    assert (response.status_code, response.content) == (
        409,
        b'{"detail":{"code":"INVALID_BINDING","edge_ref":"camera-a"}}',
    )
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == []
    assert harness.syncs == 0
    assert _registry_version(client) == 1


def test_delete_removes_camera_and_repeat_is_404(harness: Harness, client: TestClient) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a")
    harness.seed("camera-b", "rtsp://camera.example/b")

    deleted = client.delete(f"{CAMERAS}/camera-a")
    repeated = client.delete(f"{CAMERAS}/camera-a")
    unknown = client.delete(f"{CAMERAS}/missing")

    assert (deleted.status_code, deleted.content) == (204, b"")
    assert (repeated.status_code, repeated.content) == (404, NOT_FOUND)
    assert (unknown.status_code, unknown.content) == (404, NOT_FOUND)
    assert harness.audit_rows(AuditAction.CAMERA_DELETE) == [("camera-a", "admin")]
    assert harness.syncs == 1
    assert _listed_ids(client) == ["camera-b"]
    assert _registry_version(client) == 3


def test_create_then_patch_then_delete_round_trip_through_listing(
    harness: Harness, client: TestClient
) -> None:
    created = client.post(CAMERAS, json={"label": "A", "rtsp_url": "rtsp://camera.example/a"})
    camera_id = created.json()["id"]
    patched = client.patch(f"{CAMERAS}/{camera_id}", json={"label": "B", "floor": 4})
    listed = _registry(client)
    deleted = client.delete(f"{CAMERAS}/{camera_id}")

    assert (created.status_code, patched.status_code, deleted.status_code) == (201, 200, 204)
    cameras = listed["cameras"]
    assert isinstance(cameras, list)
    assert [(camera["id"], camera["label"], camera["floor"]) for camera in cameras] == [
        (camera_id, "B", 4)
    ]
    assert harness.audit_rows(AuditAction.CAMERA_CREATE) == [(camera_id, "admin")]
    assert harness.audit_rows(AuditAction.CAMERA_UPDATE) == [(camera_id, "admin")]
    assert harness.audit_rows(AuditAction.CAMERA_DELETE) == [(camera_id, "admin")]
    assert harness.syncs == 3
    assert _listed_ids(client) == []
    assert _registry_version(client) == 3


@pytest.mark.xfail(strict=True, reason="mutation responses omit mapping_state; default unmapped")
def test_update_response_reports_hub_mapping_like_listing(
    harness: Harness, client: TestClient
) -> None:
    harness.seed("camera-a", "rtsp://camera.example/a", backend_camera_id="hub-camera-a")

    response = client.patch(f"{CAMERAS}/camera-a", json={"label": "Renamed"})

    assert response.json()["backend_camera_id"] == "hub-camera-a"
    assert response.json()["mapping_state"] == "mapped"


@pytest.mark.xfail(strict=True, reason="JSON true is coerced to floor 1 before the bool guard")
def test_create_rejects_boolean_floor(harness: Harness, client: TestClient) -> None:
    response = client.post(
        CAMERAS, json={"label": "A", "rtsp_url": "rtsp://camera.example/a", "floor": True}
    )

    assert (response.status_code, response.content) == (400, b'{"detail":"invalid floor"}')
