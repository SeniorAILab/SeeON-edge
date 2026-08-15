"""Contract driver for reusable cutover mapping and backend delivery gates.

The smoke script consumes an operator-private sealed pre-cutover snapshot
and a redacted delivery readout. Expected camera count and designated
witness come from that snapshot, never from a hardcoded facility roster.
Fixtures are synthetic opaque IDs only: no production Hub, RTSP, tokens,
or host paths.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "scripts" / "ops" / "cloud-enrollment-smoke.sh"
SMOKE_TEST = REPO_ROOT / "scripts" / "ops" / "cloud-enrollment-smoke.test.sh"

LOCAL_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
LOCAL_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LOCAL_C = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
BACKEND_A = "be-aaaa1111bbbb2222cccc3333dddd4444"
BACKEND_B = "be-eeee5555ffff6666aaaa7777bbbb8888"
BACKEND_C = "be-9999cccc0000dddd1111eeee2222ffff"
WITNESS = LOCAL_C
EDGE_EVENT = "evt-aaaabbbbccccddddeeeeffff00001111"

FORBIDDEN_OUTPUT = (
    "rtsp://",
    "rtsps://",
    "password",
    "PASSWORD",
    "token=",
    "Authorization",
    "Bearer ",
    "/var/run/docker.sock",
    "--privileged",
    "Room 205",
    "hunter2",
    "super-secret-facility-token",
)
_OK = re.compile(
    r"^CUTOVER_DELIVERY_OK expected=(\d+) mapped=(\d+) heartbeats=(\d+)$",
    re.MULTILINE,
)
_FAIL = re.compile(r"^CUTOVER_DELIVERY_FAIL reason=([a-z0-9-]+)$", re.MULTILINE)
_CHECKLIST_ITEMS = (
    "enrollment",
    "sealed-pre-cutover-snapshot",
    "snapshot-derived-camera-count",
    "non-empty-backend-mapping",
    "mapping-pending-false",
    "external-heartbeat-fresh",
    "authenticated-sse-before-witness",
    "no-fabricated-event",
    "authenticated-clip",
    "authenticated-vercel-read-side",
    "edge-render-gid-matches-renderD128-owner",
    "no-repository-gid-default",
    "no-socket-privileged-docker-group-bypass",
    "legacy-rollback-boundary-preserved",
)


def _sha256_text(payload: str) -> str:
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _camera(
    local_id: str,
    *,
    witness: bool = False,
) -> dict[str, str]:
    return {
        "localId": local_id,
        "class": "expected-witness" if witness else "expected",
    }


def _snapshot(
    *,
    cameras: list[dict[str, str]] | None = None,
    camera_count: int | None = None,
    witness: str = WITNESS,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = cameras or [
        _camera(LOCAL_A),
        _camera(LOCAL_B),
        _camera(LOCAL_C, witness=True),
    ]
    body: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "pre-cutover-snapshot",
        "cameraCount": len(rows) if camera_count is None else camera_count,
        "cameras": rows,
        "designatedWitnessLocalId": witness,
    }
    if extra:
        body.update(extra)
    return body


def _delivery_camera(
    local_id: str,
    backend_id: str,
    *,
    mapping_pending: bool = False,
    heartbeat_ok: bool = True,
    heartbeat_fresh: bool = True,
) -> dict[str, Any]:
    return {
        "localId": local_id,
        "backendCameraId": backend_id,
        "mappingPending": mapping_pending,
        "externalHeartbeat": {"ok": heartbeat_ok, "fresh": heartbeat_fresh},
    }


def _delivery(
    *,
    cameras: list[dict[str, Any]] | None = None,
    enrollment_ok: bool = True,
    enrollment_auth: bool = True,
    sse_auth: bool = True,
    sse_at: str = "2026-08-11T00:00:00Z",
    sse_before: bool = True,
    witness: str = WITNESS,
    processing_at: str = "2026-08-11T00:00:02Z",
    fabricated: bool = False,
    real_event: bool = True,
    edge_event_id: str = EDGE_EVENT,
    clip_auth: bool = True,
    vercel_auth: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "cutover-delivery-readout",
        "enrollment": {"ok": enrollment_ok, "authenticated": enrollment_auth},
        "cameras": cameras
        or [
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_B),
            _delivery_camera(LOCAL_C, BACKEND_C),
        ],
        "sse": {
            "authenticated": sse_auth,
            "establishedAt": sse_at,
            "establishedBeforeWitnessProcessing": sse_before,
        },
        "witness": {
            "localId": witness,
            "processingEnabledAt": processing_at,
            "eventFabricated": fabricated,
            "realEvent": real_event,
            "edgeEventId": edge_event_id,
            "clipAuthenticated": clip_auth,
            "vercelReadSideAuthenticated": vercel_auth,
        },
    }
    if extra:
        body.update(extra)
    return body


def _write_json(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assert_redacted(result: subprocess.CompletedProcess[str], *extra: str) -> None:
    blob = f"{result.stdout}\n{result.stderr}"
    for needle in (*FORBIDDEN_OUTPUT, *extra):
        assert needle not in blob, f"unredacted {needle!r} in script output"


def _run(
    tmp_path: Path,
    *,
    args: tuple[str, ...] = (),
    env_extra: Mapping[str, str] | None = None,
    snapshot: dict[str, Any] | None = None,
    delivery: dict[str, Any] | None = None,
    snapshot_sha: str | None = None,
    delivery_sha: str | None = None,
    mutate_snapshot_after_hash: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("EDGE_PROVISIONING_SNAPSHOT", None)
    env.pop("EDGE_PROVISIONING_SNAPSHOT_SHA256", None)
    env.pop("EDGE_PROVISIONING_DELIVERY", None)
    env.pop("EDGE_PROVISIONING_DELIVERY_SHA256", None)
    if env_extra:
        env.update(env_extra)
    cmd = ["sh", str(SMOKE), *args]
    if snapshot is not None:
        snap_path = _write_json(tmp_path / "snapshot.json", snapshot)
        digest = _sha256_text(snap_path.read_text(encoding="utf-8"))
        if mutate_snapshot_after_hash:
            mutated = json.loads(snap_path.read_text(encoding="utf-8"))
            mutated["cameraCount"] = int(mutated["cameraCount"]) + 1
            _write_json(snap_path, mutated)
        env["EDGE_PROVISIONING_SNAPSHOT"] = str(snap_path)
        env["EDGE_PROVISIONING_SNAPSHOT_SHA256"] = snapshot_sha or digest
        cmd += ["--snapshot", str(snap_path)]
    if delivery is not None:
        del_path = _write_json(tmp_path / "delivery.json", delivery)
        digest = _sha256_text(del_path.read_text(encoding="utf-8"))
        env["EDGE_PROVISIONING_DELIVERY"] = str(del_path)
        env["EDGE_PROVISIONING_DELIVERY_SHA256"] = delivery_sha or digest
        cmd += ["--delivery", str(del_path)]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )


def test_smoke_script_is_executable_operator_interface() -> None:
    assert SMOKE.is_file()
    assert SMOKE.stat().st_mode & stat.S_IXUSR
    assert SMOKE_TEST.is_file()
    assert SMOKE_TEST.stat().st_mode & stat.S_IXUSR


def test_print_checklist_requires_host_derived_render_gid() -> None:
    result = subprocess.run(
        ["sh", str(SMOKE), "--print-checklist"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CUTOVER_OPERATOR_CHECKLIST" in result.stdout
    for item in _CHECKLIST_ITEMS:
        assert f"item={item}" in result.stdout
    assert "EDGE_RENDER_GID" in result.stdout
    assert "renderD128" in result.stdout
    assert "104" not in result.stdout
    assert "44" not in result.stdout
    _assert_redacted(result)


def test_all_mapped_fresh_heartbeats_pass(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    match = _OK.search(result.stdout)
    assert match is not None, result.stdout
    assert match.group(1) == match.group(2) == match.group(3) == "3"
    assert "13" not in result.stdout
    assert "Room 205" not in result.stdout
    _assert_redacted(result)


def test_two_local_cameras_sharing_backend_id_is_duplicate_mapping(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        cameras=[_camera(LOCAL_A), _camera(LOCAL_B, witness=True)],
        witness=LOCAL_B,
    )
    delivery = _delivery(
        cameras=[
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_A),
        ],
        witness=LOCAL_B,
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=snapshot,
        delivery=delivery,
    )
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "CUTOVER_DELIVERY_OK" not in blob
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None, blob
    assert match.group(1) == "duplicate-mapping"
    _assert_redacted(result, BACKEND_A)


def test_print_checklist_and_delivery_gate_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        args=("--print-checklist", "--delivery-gate"),
        snapshot=_snapshot(),
        delivery=_delivery(),
    )
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "CUTOVER_DELIVERY_OK" not in blob
    _assert_redacted(result)


def test_one_unmapped_camera_is_a_clear_stop(tmp_path: Path) -> None:
    delivery = _delivery(
        cameras=[
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_B),
            _delivery_camera(LOCAL_C, ""),
        ]
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=delivery,
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None, result.stdout + result.stderr
    assert match.group(1) in {"unmapped-camera", "blank-mapping"}
    _assert_redacted(result)


def test_mapping_pending_true_is_a_clear_stop(tmp_path: Path) -> None:
    delivery = _delivery(
        cameras=[
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_B, mapping_pending=True),
            _delivery_camera(LOCAL_C, BACKEND_C),
        ]
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=delivery,
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "mapping-pending"
    _assert_redacted(result)


def test_one_missed_heartbeat_is_a_clear_stop(tmp_path: Path) -> None:
    delivery = _delivery(
        cameras=[
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_B, heartbeat_ok=False),
            _delivery_camera(LOCAL_C, BACKEND_C),
        ]
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=delivery,
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "missed-heartbeat"
    _assert_redacted(result)


def test_stale_heartbeat_is_a_clear_stop(tmp_path: Path) -> None:
    delivery = _delivery(
        cameras=[
            _delivery_camera(LOCAL_A, BACKEND_A),
            _delivery_camera(LOCAL_B, BACKEND_B, heartbeat_fresh=False),
            _delivery_camera(LOCAL_C, BACKEND_C),
        ]
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=delivery,
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "stale-heartbeat"
    _assert_redacted(result)


def test_count_drift_between_snapshot_and_delivery_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(camera_count=4),
        delivery=_delivery(),
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "count-drift"
    _assert_redacted(result)


def test_sse_after_witness_processing_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(
            sse_at="2026-08-11T00:00:03Z",
            processing_at="2026-08-11T00:00:02Z",
            sse_before=True,
        ),
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "sse-order"
    _assert_redacted(result)


def test_unauthenticated_sse_or_enrollment_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(sse_auth=False),
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "auth-absent"
    _assert_redacted(result)


def test_fabricated_event_is_rejected(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(fabricated=True, real_event=False),
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "fabricated-event"
    _assert_redacted(result)


def test_unauthenticated_clip_or_vercel_fails(tmp_path: Path) -> None:
    clip = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(clip_auth=False),
    )
    assert clip.returncode != 0
    clip_match = _FAIL.search(clip.stdout) or _FAIL.search(clip.stderr)
    assert clip_match is not None
    assert clip_match.group(1) == "clip-unauthenticated"
    _assert_redacted(clip)

    other = tmp_path / "vercel"
    other.mkdir()
    vercel = _run(
        other,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(vercel_auth=False),
    )
    assert vercel.returncode != 0
    vercel_match = _FAIL.search(vercel.stdout) or _FAIL.search(vercel.stderr)
    assert vercel_match is not None
    assert vercel_match.group(1) == "vercel-unauthenticated"
    _assert_redacted(vercel)


@pytest.mark.parametrize(
    "bad_id",
    (
        "Room 205",
        "room-205",
        "",
        "   ",
        "cam/1",
        "rtsp://operator:secret@192.0.2.10/live",
        "ab",
    ),
)
def test_malformed_opaque_ids_fail_closed(tmp_path: Path, bad_id: str) -> None:
    snapshot = _snapshot(
        cameras=[_camera(LOCAL_A), _camera(LOCAL_B), _camera(bad_id, witness=True)],
        witness=bad_id or LOCAL_C,
    )
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=snapshot,
        delivery=_delivery(witness=bad_id or LOCAL_C),
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) in {"malformed-id", "malformed-receipt", "secret-bearing"}
    _assert_redacted(result, bad_id if "secret" in bad_id else "unused-sentinel")


def test_secret_bearing_receipt_is_rejected_and_redacted(tmp_path: Path) -> None:
    delivery = _delivery(extra={"password": "hunter2", "token": "super-secret-facility-token"})
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=delivery,
    )
    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "secret-bearing"
    _assert_redacted(result)


def test_dirty_snapshot_after_hash_is_rejected(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(),
        mutate_snapshot_after_hash=True,
    )
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "CUTOVER_DELIVERY_OK" not in blob
    _assert_redacted(result)


def test_missing_snapshot_or_delivery_is_a_clear_stop(tmp_path: Path) -> None:
    result = _run(tmp_path, args=("--delivery-gate",))
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "CUTOVER_DELIVERY_OK" not in blob
    assert "snapshot" in blob.lower() or "delivery" in blob.lower()
    _assert_redacted(result)


def test_delivery_gate_does_not_call_a_production_hub(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("curl", "ssh", "nc"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\necho contacted-network >&2\nexit 97\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = _run(
        tmp_path,
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(),
        env_extra=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "contacted-network" not in result.stdout + result.stderr
    _assert_redacted(result)


def test_fixture_covers_mapped_unmapped_heartbeat_and_gid_checklist() -> None:
    result = subprocess.run(
        ["sh", str(SMOKE), "--fixture", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    for token in (
        "EDGE_EXECUTION_CONTENT_ADDRESS_REJECTION_OK",
        "EDGE_IMAGE_PROVENANCE_REJECTION_OK",
        "EDGE_DUPLICATE_HOST_KEY_REJECTION_OK",
        "EDGE_DELIVERY_ALL_MAPPED_OK",
        "EDGE_DELIVERY_UNMAPPED_REJECTION_OK",
        "EDGE_DELIVERY_MISSED_HEARTBEAT_REJECTION_OK",
        "EDGE_DELIVERY_DUPLICATE_MAPPING_REJECTION_OK",
        "EDGE_CHECKLIST_DELIVERY_EXCLUSIVE_OK",
        "CUTOVER_CHECKLIST_GID_CONTRACT_OK",
        "CLOUD_ENROLLMENT_SMOKE_FIXTURE_OK",
    ):
        assert token in result.stdout, token
    _assert_redacted(result)


def test_existing_spoof_harness_still_rejects_attacker_receipts() -> None:
    result = subprocess.run(
        ["sh", str(SMOKE_TEST)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "CLOUD_ENROLLMENT_SPOOF_REJECTION_OK" in result.stdout
    _assert_redacted(result)


def test_fixture_and_delivery_gate_clean_up_temp_dirs(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    before = set(tmp_path.iterdir())
    fixture = subprocess.run(
        ["sh", str(SMOKE), "--fixture", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert fixture.returncode == 0, fixture.stderr
    delivery = _run(
        tmp_path / "gate",
        args=("--delivery-gate",),
        snapshot=_snapshot(),
        delivery=_delivery(),
        env_extra=env,
    )
    assert delivery.returncode == 0, delivery.stderr
    leftover = [
        path
        for path in tmp_path.rglob("*")
        if path.is_dir()
        and path.name.startswith(("cloud-enrollment-smoke", "cutover-delivery"))
    ]
    assert leftover == []
    assert set(tmp_path.iterdir()) >= before
    _assert_redacted(fixture)
    _assert_redacted(delivery)


def test_no_socket_or_docker_group_bypass_in_gate_surface() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")
    assert "docker.sock" not in smoke
    assert "--privileged" not in smoke
    assert "usermod" not in smoke
    assert "-G docker" not in smoke
