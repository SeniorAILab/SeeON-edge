"""Characterization of the existing env-file preflight and inventory gate.

These tests pin today's check-env.sh behavior (inventory, overlay selection,
HTTPS/dashboard/RTSP gates) before the root-owned carrier and required GID
bindings are added. They never talk to a live Docker daemon.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from backend.app.features.connection.hub_url import UNSUPPORTED_HUB_API_BASE_PATH_REASON

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "scripts" / "edge-preflight" / "check-env.sh"
INVENTORY = REPO_ROOT / "edge-env-inventory.json"
EXAMPLE = REPO_ROOT / ".env.edge.prod.example"

_BASE_ENV = "\n".join(
    (
        "ML_API_IMAGE=ghcr.io/example/ml-api@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ML_WORKER_IMAGE=ghcr.io/example/ml-worker@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "ML_WORKER_PROFILE=cpu-host",
        "CLIP_STORE_HOST_DIR=/tmp/clip-store-preflight",
        "API_BACKEND_BASE_URL=https://hub.example.test",
        "API_BACKEND_INGEST_TIMEOUT_SEC=10",
        "API_DASHBOARD_USERNAME=site-ops-contract",
        "API_DASHBOARD_PASSWORD=disposable-bootstrap-9f3a",
        "API_EDGE_RELAY_TOKEN=disposable-relay-7c1e5b9a2f4d8e6b",
        "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS=1",
        "ML_RTSP_ALLOW_LOCAL_DESTINATIONS=0",
        "",
    )
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _merge_env(content: str) -> str:
    def _parse(block: str) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            rows.append((key, key.upper(), value))
        return rows

    base_rows = _parse(_BASE_ENV)
    content_rows = _parse(content)
    content_keys = {normalized for _, normalized, _ in content_rows}
    merged: list[str] = []
    for key, normalized, value in base_rows:
        if normalized not in content_keys:
            merged.append(f"{key}={value}")
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            merged.append(raw)
            continue
        key, value = stripped.split("=", 1)
        merged.append(f"{key}={value}")
    return "\n".join(merged) + "\n"


def _run_preflight(
    tmp_path: Path,
    content: str,
    *compose_args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / "edge.env"
    env_file.write_text(_merge_env(content), encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    _write_executable(
        bin_dir / "docker",
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "${FAKE_DOCKER_LOG:?}"\n'
        'if [ "${1:-}" = compose ] && [ "${2:-}" = version ]; then exit 0; fi\n'
        "exit 0\n",
    )
    env = {
        **os.environ,
        "FAKE_DOCKER_LOG": str(docker_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    env.pop("EDGE_RENDER_GID", None)
    env.pop("EDGE_VIDEO_GID", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(PREFLIGHT), str(env_file), *compose_args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _render_invocations(tmp_path: Path) -> list[str]:
    log = tmp_path / "docker.log"
    if not log.is_file():
        return []
    return [line for line in log.read_text(encoding="utf-8").splitlines() if " config" in line]


def test_preflight_accepts_inventory_complete_cpu_env(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert "passes inventory, compose, Hub HTTPS, dashboard, and RTSP gates" in result.stdout
    invocations = _render_invocations(tmp_path)
    assert invocations
    assert all("compose.edge.yaml" in line for line in invocations)
    assert all("compose.edge.igpu.yaml" not in line for line in invocations)
    assert all("compose.edge.nvidia.yaml" not in line for line in invocations)


@pytest.mark.parametrize(
    ("profile", "overlay"),
    [
        ("cpu-host", None),
        ("intel-vaapi-host", "compose.edge.igpu.yaml"),
        ("igpu", "compose.edge.igpu.yaml"),
        ("nvidia-host-bridge", "compose.edge.nvidia.yaml"),
    ],
)
def test_preflight_selects_profile_overlay(
    tmp_path: Path, profile: str, overlay: str | None
) -> None:
    extra = f"ML_WORKER_PROFILE={profile}\n"
    extra_env = None
    if overlay == "compose.edge.igpu.yaml":
        extra += "EDGE_RENDER_GID=993\nEDGE_VIDEO_GID=44\n"
        device = tmp_path / "renderD128"
        device.write_bytes(b"")
        extra_env = {
            "EDGE_RENDER_DEVICE": str(device),
            "EDGE_RENDER_DEVICE_GID": "993",
        }
    result = _run_preflight(tmp_path, extra, extra_env=extra_env)
    assert result.returncode == 0, result.stderr
    invocations = _render_invocations(tmp_path)
    assert invocations
    if overlay is None:
        assert all("compose.edge.igpu.yaml" not in line for line in invocations)
        assert all("compose.edge.nvidia.yaml" not in line for line in invocations)
    else:
        assert all(overlay in line for line in invocations)


def test_preflight_rejects_unregistered_environment_key(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "EDGE_UNKNOWN_BINDING=99\n")
    assert result.returncode != 0
    assert "EDGE_UNKNOWN_BINDING" in result.stderr
    assert "unsupported key" in result.stderr
    assert _render_invocations(tmp_path) == []


def test_preflight_rejects_retired_key_before_compose(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "ML_SERVING_PORT=8000\n")
    assert result.returncode != 0
    assert "ML_SERVING_PORT" in result.stderr
    assert "retired" in result.stderr
    assert _render_invocations(tmp_path) == []


def test_preflight_rejects_malformed_assignment(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "not a key value line\n")
    assert result.returncode != 0
    assert "malformed env assignment" in result.stderr
    assert _render_invocations(tmp_path) == []


def test_preflight_rejects_cleartext_hub_and_redacts_nothing_secret(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "API_BACKEND_BASE_URL=http://hub.example.test\n")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "https://" in combined
    assert "disposable-bootstrap-9f3a" not in combined
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in combined


@pytest.mark.parametrize(
    "base",
    [
        "https://hub.example.test",
        "https://hub.example.test/",
        "https://hub.example.test/api",
        "https://hub.example.test/api/",
    ],
)
def test_preflight_accepts_origin_and_optional_api_bases(
    tmp_path: Path, base: str
) -> None:
    result = _run_preflight(tmp_path, f"API_BACKEND_BASE_URL={base}\n")
    assert result.returncode == 0, result.stderr
    assert "passes inventory, compose, Hub HTTPS, dashboard, and RTSP gates" in result.stdout
    assert "disposable-bootstrap-9f3a" not in result.stdout + result.stderr
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "base",
    [
        "https://hub.example.test/api/v1",
        "https://hub.example.test/api/v1/",
        "https://hub.example.test/foo",
        "https://hub.example.test/api/v1/events",
        "https://hub.example.test/api?x=1",
        "https://hub.example.test/api#frag",
        "https://user:pass@hub.example.test",
        "https://hub.example.test/api/../api",
        "https://hub.example.test/API",
        "https://hub.example.test /api",
        "https://hub.example.test/api/v1?x=1",
        "https://hub.example.test/api/./v1",
    ],
)
def test_preflight_rejects_unsupported_hub_api_paths(
    tmp_path: Path, base: str
) -> None:
    result = _run_preflight(tmp_path, f"API_BACKEND_BASE_URL={base}\n")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert UNSUPPORTED_HUB_API_BASE_PATH_REASON in combined
    assert "disposable-bootstrap-9f3a" not in combined
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in combined
    assert "passes inventory, compose, Hub HTTPS, dashboard, and RTSP gates" not in result.stdout


def test_preflight_hub_path_uses_env_file_not_process_env(tmp_path: Path) -> None:
    accepted = _run_preflight(
        tmp_path / "accepted",
        "API_BACKEND_BASE_URL=https://hub.example.test\n",
        extra_env={"API_BACKEND_BASE_URL": "https://hub.example.test/api/v1"},
    )
    assert accepted.returncode == 0, accepted.stderr
    rejected = _run_preflight(
        tmp_path / "rejected",
        "API_BACKEND_BASE_URL=https://hub.example.test/api/v1\n",
        extra_env={"API_BACKEND_BASE_URL": "https://hub.example.test"},
    )
    assert rejected.returncode != 0
    assert UNSUPPORTED_HUB_API_BASE_PATH_REASON in rejected.stdout + rejected.stderr


def test_preflight_rejects_fixture_only_local_rtsp_flag(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "ML_RTSP_ALLOW_LOCAL_DESTINATIONS=1\n")
    assert result.returncode != 0
    assert "ML_RTSP_ALLOW_LOCAL_DESTINATIONS" in result.stderr


GID_BINDINGS = ("EDGE_RENDER_GID", "EDGE_VIDEO_GID")


def _gid_env(
    *,
    render: str = "993",
    video: str = "44",
    profile: str = "intel-vaapi-host",
    extra: str = "",
) -> str:
    return (
        f"ML_WORKER_PROFILE={profile}\n"
        f"EDGE_RENDER_GID={render}\n"
        f"EDGE_VIDEO_GID={video}\n"
        f"{extra}"
    )


def _synthetic_render_device(tmp_path: Path, gid: int) -> Path:
    device = tmp_path / "renderD128"
    device.write_bytes(b"")
    try:
        os.chown(device, os.getuid(), gid)
    except (OSError, PermissionError):
        # macOS tests cannot chown to an arbitrary GID. The preflight must
        # still accept an explicit expected-GID override for the fixture.
        pass
    return device


def test_inventory_registers_gid_bindings_without_host_defaults() -> None:
    import json

    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in payload["variables"]}
    example = EXAMPLE.read_text(encoding="utf-8")
    assigned = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    commented = set(re.findall(r"^#\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    for name in GID_BINDINGS:
        entry = by_name[name]
        assert entry["compose"] is True
        assert entry["example"] is True
        assert entry["category"] == "deployment artifact"
        assert "default" not in entry["behavior"].lower() or "no host default" in entry["behavior"]
        assert name not in assigned
        assert name in commented
    assert "104" not in example
    assert "getent" in example.lower() or "host" in example.lower()


def test_preflight_accepts_registered_matching_gids(tmp_path: Path) -> None:
    device = _synthetic_render_device(tmp_path, 993)
    result = _run_preflight(
        tmp_path,
        _gid_env(),
        extra_env={
            "EDGE_RENDER_DEVICE": str(device),
            "EDGE_RENDER_DEVICE_GID": "993",
        },
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "disposable-bootstrap-9f3a" not in combined
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in combined


@pytest.mark.parametrize("missing", GID_BINDINGS)
def test_preflight_rejects_missing_gid_for_igpu_profile(
    tmp_path: Path, missing: str
) -> None:
    extra = _gid_env()
    extra = "\n".join(
        line for line in extra.splitlines() if not line.startswith(f"{missing}=")
    )
    result = _run_preflight(tmp_path, extra + "\n")
    assert result.returncode != 0
    assert missing in result.stderr
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "-1",
        "12.5",
        "0x2c",
        "44;id",
        "44$(id)",
        "44`id`",
        "render",
        "104,44",
        "999999999999",
    ],
)
def test_preflight_rejects_malformed_gid_strings(tmp_path: Path, raw: str) -> None:
    result = _run_preflight(tmp_path, _gid_env(render=raw))
    assert result.returncode != 0
    assert "EDGE_RENDER_GID" in result.stderr
    assert raw.strip() == "" or raw.split(";", 1)[0][:8] in result.stderr or "GID" in result.stderr
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in result.stderr
    assert _render_invocations(tmp_path) == []


def test_preflight_rejects_host_device_gid_mismatch(tmp_path: Path) -> None:
    device = _synthetic_render_device(tmp_path, 44)
    result = _run_preflight(
        tmp_path,
        _gid_env(render="993", video="44"),
        extra_env={
            "EDGE_RENDER_DEVICE": str(device),
            "EDGE_RENDER_DEVICE_GID": "44",
        },
    )
    assert result.returncode != 0
    assert "EDGE_RENDER_GID" in result.stderr
    assert "mismatch" in result.stderr.lower()
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in result.stderr


def test_preflight_rejects_absent_render_device_for_igpu(tmp_path: Path) -> None:
    result = _run_preflight(
        tmp_path,
        _gid_env(),
        extra_env={"EDGE_RENDER_DEVICE": str(tmp_path / "missing-renderD128")},
    )
    assert result.returncode != 0
    assert "render" in result.stderr.lower()
    missing_device = str(tmp_path / "missing-renderD128")
    assert missing_device not in result.stderr


def test_preflight_rejects_process_env_gid_override(tmp_path: Path) -> None:
    device = _synthetic_render_device(tmp_path, 993)
    result = _run_preflight(
        tmp_path,
        _gid_env(render="993", video="44"),
        extra_env={
            "EDGE_RENDER_GID": "0",
            "EDGE_VIDEO_GID": "0",
            "EDGE_RENDER_DEVICE": str(device),
            "EDGE_RENDER_DEVICE_GID": "993",
        },
    )
    assert result.returncode != 0
    assert "process" in result.stderr.lower() or "override" in result.stderr.lower()
    assert "EDGE_RENDER_GID" in result.stderr


def test_preflight_cpu_profile_does_not_require_gids(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "ML_WORKER_PROFILE=cpu-host\n")
    assert result.returncode == 0, result.stderr


LEGACY_VOLUME_BINDINGS = (
    "EDGE_LEGACY_CATALOG_VOLUME",
    "EDGE_LEGACY_CONNECTION_VOLUME",
    "EDGE_LEGACY_WORKER_VOLUME",
)
_SYNTHETIC_LEGACY_VOLUMES = {
    "EDGE_LEGACY_CATALOG_VOLUME": "seeon-contract-legacy-catalog",
    "EDGE_LEGACY_CONNECTION_VOLUME": "seeon-contract-legacy-connection",
    "EDGE_LEGACY_WORKER_VOLUME": "seeon-contract-legacy-worker",
}


def _legacy_env() -> str:
    return "".join(f"{key}={value}\n" for key, value in _SYNTHETIC_LEGACY_VOLUMES.items())


def test_inventory_registers_cutover_volume_bindings_without_host_defaults() -> None:
    import json

    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in payload["variables"]}
    example = EXAMPLE.read_text(encoding="utf-8")
    assigned = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    commented = set(re.findall(r"^#\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    for name in LEGACY_VOLUME_BINDINGS:
        entry = by_name[name]
        assert entry["compose"] is True
        assert entry["example"] is True
        assert entry["category"] == "deployment artifact"
        assert "compose.edge.migrate.yaml" in entry["behavior"]
        assert "default" not in entry["behavior"].lower() or "no host default" in entry[
            "behavior"
        ]
        assert name not in assigned
        assert name in commented
    for token in ("happy", "nursing", "hn-", "COMPOSE_PROJECT_NAME"):
        assert token not in example.lower()


def _run_real_preflight(
    tmp_path: Path,
    content: str,
    *compose_args: str,
) -> subprocess.CompletedProcess[str]:
    env_file = tmp_path / "edge.env"
    env_file.write_text(_merge_env(content), encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"EDGE_RENDER_GID", "EDGE_VIDEO_GID"}
    }
    return subprocess.run(
        [str(PREFLIGHT), str(env_file), *compose_args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_preflight_accepts_registered_cutover_volume_bindings(tmp_path: Path) -> None:
    result = _run_real_preflight(
        tmp_path,
        _legacy_env(),
        "-f",
        "compose.edge.migrate.yaml",
    )
    assert result.returncode == 0, result.stderr
    assert "passes inventory, compose, Hub HTTPS, dashboard, and RTSP gates" in result.stdout
    combined = result.stdout + result.stderr
    assert "disposable-bootstrap-9f3a" not in combined
    assert "disposable-relay-7c1e5b9a2f4d8e6b" not in combined


def test_preflight_rejects_cutover_overlay_without_legacy_bindings(
    tmp_path: Path,
) -> None:
    result = _run_real_preflight(tmp_path, "", "-f", "compose.edge.migrate.yaml")
    assert result.returncode != 0
    assert "EDGE_LEGACY_" in result.stderr
    assert "required variable" in result.stderr or "cutover requires" in result.stderr
    assert result.stdout.strip() == ""
