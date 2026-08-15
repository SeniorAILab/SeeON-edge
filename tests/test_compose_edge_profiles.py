from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from backend.app.core.config import reject_retired_backend_environment
from backend.app.features.connection import store as connection_store_module
from backend.app.lifespan import (
    InvalidBackendIngestTimeoutError,
    _backend_ingest_timeout_sec,
)
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config
from worker.runtime.config.local_env import reject_retired_worker_environment
from worker.runtime.config.pull_models import BackendWorkerConfigPayload

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "edge-env-inventory.json"

EXTERNAL_COMPOSE_KEYS = {
    "API_BACKEND_BASE_URL",
    "API_BACKEND_INGEST_TIMEOUT_SEC",
    "API_DASHBOARD_PASSWORD",
    "API_DASHBOARD_USERNAME",
    "API_EDGE_RELAY_TOKEN",
    "CLIP_STORE_HOST_DIR",
    "ML_API_IMAGE",
    "ML_RTSP_ALLOW_LOCAL_DESTINATIONS",
    "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS",
    "ML_WORKER_IMAGE",
    "ML_WORKER_PROFILE",
}
MIGRATION_OVERLAY_FILE = ROOT / "compose.edge.migrate.yaml"
LEGACY_VOLUME_BINDINGS = {
    "EDGE_LEGACY_CATALOG_VOLUME",
    "EDGE_LEGACY_CONNECTION_VOLUME",
    "EDGE_LEGACY_WORKER_VOLUME",
}
_LEGACY_VOLUME_MARKERS = ("ml-api-state", "ml-worker-state")
HISTORICAL_COMPOSE_EXAMPLE_KEYS = EXTERNAL_COMPOSE_KEYS | {
    "API_ALLOW_LEGACY_DASHBOARD_AUTH",
    "API_BACKEND_CONFIG_URL",
    "API_BACKEND_EVENTS_URL",
    "ML_API_DETECTION_TZ",
    "ML_API_EVENT_CLIP_EXPORT_ENABLED",
    "ML_DEFAULT_CAMERA_FPS",
    "ML_DEFAULT_FRAME_STRIDE",
    "ML_MODELS_DIR",
    "ML_SERVING_PORT",
    "ML_WORKER_CLIP_RECORDING_ENABLED",
    "ML_WORKER_DEV_MJPEG",
    "ML_WORKER_DEV_MJPEG_PORT",
    "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED",
    "ML_WORKER_FALL_MODEL_ARTIFACT_DIR",
    "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD",
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY",
    "ML_WORKER_FALL_MODEL_SCHEMA_VERSION",
    "ML_WORKER_FALL_MODEL_STRIDE",
    "ML_WORKER_FALL_MODEL_WINDOW",
}


def _compose() -> dict[str, object]:
    class Loader(yaml.SafeLoader):
        pass

    def compose_tag(loader: Loader, _suffix: str, node: yaml.Node) -> object:
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return None

    Loader.add_multi_constructor("!", compose_tag)
    payload = yaml.load((ROOT / "compose.edge.yaml").read_text(), Loader=Loader)
    assert isinstance(payload, dict)
    return payload


def _inventory() -> dict[str, object]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_machine_inventory_accounts_for_every_historical_compose_example_key() -> None:
    inventory = _inventory()
    entries = inventory["variables"]
    assert isinstance(entries, list)
    by_name = {entry["name"]: entry for entry in entries}

    assert HISTORICAL_COMPOSE_EXAMPLE_KEYS <= set(by_name)
    assert all(
        entry["category"]
        in {
            "deployment artifact",
            "secret/bootstrap",
            "fixed internal topology",
            "mutable policy",
            "legacy compatibility",
            "dead",
        }
        for entry in entries
    )
    assert all(entry["authority"] and entry["behavior"] for entry in entries)


def test_compose_and_example_external_keys_cannot_drift_from_inventory() -> None:
    compose_text = (ROOT / "compose.edge.yaml").read_text(encoding="utf-8")
    compose_keys = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose_text))
    example_keys = set(
        re.findall(
            r"^([A-Z][A-Z0-9_]*)=",
            (ROOT / ".env.edge.prod.example").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    inventory = _inventory()
    entries = inventory["variables"]
    assert isinstance(entries, list)
    exposed = {
        entry["name"]
        for entry in entries
        if entry.get("compose") is True or entry.get("example") is True
    }
    igpu_keys = set(
        re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)",
            (ROOT / "compose.edge.igpu.yaml").read_text(encoding="utf-8"),
        )
    )

    assert compose_keys == EXTERNAL_COMPOSE_KEYS
    assert example_keys == EXTERNAL_COMPOSE_KEYS
    assert igpu_keys >= {"EDGE_RENDER_GID", "EDGE_VIDEO_GID"}
    migrate_keys = set(
        re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)",
            MIGRATION_OVERLAY_FILE.read_text(encoding="utf-8"),
        )
    )
    assert migrate_keys == LEGACY_VOLUME_BINDINGS
    assert (
        compose_keys
        | example_keys
        | {"EDGE_RENDER_GID", "EDGE_VIDEO_GID"}
        | LEGACY_VOLUME_BINDINGS
        == exposed
    )

    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    service_environment = {
        key
        for service in services.values()
        if isinstance(service, dict)
        for key in service.get("environment", {})
    }
    inventoried_service_environment = {
        entry["name"] for entry in entries if entry.get("service_environment") is True
    }
    assert service_environment == inventoried_service_environment


def test_compose_profile_topology_is_nvidia_opt_in() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    worker = services["ml-worker"]
    assert isinstance(worker, dict)
    assert "deploy" not in worker
    worker_env = worker["environment"]
    assert isinstance(worker_env, dict)
    assert "NVIDIA_DRIVER_CAPABILITIES" not in worker_env

    nvidia_text = (ROOT / "compose.edge.nvidia.yaml").read_text(encoding="utf-8")
    assert "driver: nvidia" in nvidia_text
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility,video" in nvidia_text
    assert "cpu-host" in (ROOT / ".env.edge.prod.example").read_text(encoding="utf-8")
    assert "nvidia-device-experimental" in nvidia_text


def test_inventory_records_completed_profile_canonicalization() -> None:
    profile_entry = next(
        entry for entry in _inventory()["variables"] if entry["name"] == "ML_WORKER_PROFILE"
    )
    assert profile_entry["todo7_status"] == "implemented"
    assert profile_entry["canonical_choices"] == [
        "cpu-host",
        "nvidia-host-bridge",
        "intel-vaapi-host",
        "apple-mps-host",
        "nvidia-device-experimental",
    ]


def test_compose_bakes_internal_topology_and_has_no_policy_or_state_path_env() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    api = services["ml-api"]
    worker = services["ml-worker"]
    assert isinstance(api, dict)
    assert isinstance(worker, dict)
    api_env = api["environment"]
    worker_env = worker["environment"]
    assert isinstance(api_env, dict)
    assert isinstance(worker_env, dict)

    assert api["ports"] == ["127.0.0.1:8000:8000"]
    assert "ML_API_WORKER_STREAM_ORIGIN" not in api_env
    assert "ML_API_WORKER_PROBE_ORIGIN" not in api_env
    assert "RELAY_URL" not in worker_env
    assert "ML_WORKER_DEV_MJPEG" not in worker_env
    assert "ML_WORKER_DEV_MJPEG_HOST" not in worker_env
    assert "ML_WORKER_DEV_MJPEG_PORT" not in worker_env
    assert not any("STATE" in key or "POLICY" in key for key in api_env | worker_env)


def test_retired_backend_environment_is_rejected_explicitly() -> None:
    for key in (
        "API_FACILITY_ID",
        "EDGE_FACILITY_TOKEN",
        "API_CAMERA_INVENTORY",
        "API_BACKEND_EVENTS_URL",
        "API_BACKEND_CONFIG_URL",
        "ML_DEFAULT_CAMERA_FPS",
        "ML_API_DETECTION_TZ",
    ):
        with pytest.raises(ValueError, match=key):
            reject_retired_backend_environment({key: "injected"})


def test_retired_backend_environment_normalizes_case_like_settings() -> None:
    with pytest.raises(ValueError, match="ml_api_worker_stream_origin"):
        reject_retired_backend_environment({"ml_api_worker_stream_origin": "http://wrong.example"})


def test_retired_worker_environment_is_rejected_explicitly() -> None:
    for key in (
        "ML_WORKER_FALL_MODEL_WINDOW",
        "ML_WORKER_CLIP_RECORDING_ENABLED",
        "ML_WORKER_DEV_MJPEG_PORT",
        "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED",
    ):
        with pytest.raises(WorkerConfigError, match=key):
            reject_retired_worker_environment({key: "injected"})


def test_static_yaml_camera_roster_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "relay:\n  url: http://ml-api:8000\n  token: relay-token\n"
        "cameras:\n  - camera_id: camera-1\n    facility_id: facility-1\n"
        "    rtsp_url: rtsp://camera/live\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkerConfigError, match="static camera roster is retired"):
        load_worker_config(config_path)


@pytest.mark.parametrize("field", ["models", "domains", "clip"])
def test_static_yaml_mutable_policy_is_rejected(tmp_path: Path, field: str) -> None:
    config_path = tmp_path / "worker.yaml"
    values = {
        "models": "models: {}\n",
        "domains": "domains: {}\n",
        "clip": "clip:\n  enabled: false\n",
    }
    config_path.write_text(
        "relay:\n  url: http://ml-api:8000\n  token: relay-token\n" + values[field],
        encoding="utf-8",
    )

    with pytest.raises(WorkerConfigError, match=f"static {field} policy is retired"):
        load_worker_config(config_path)


def test_pull_payload_rejects_unversioned_model_or_clip_policy() -> None:
    for field, value in (("models", {}), ("clip", {"enabled": False})):
        with pytest.raises(ValueError, match=field):
            BackendWorkerConfigPayload.model_validate(
                {"config_version": 4, "cameras": [], field: value}
            )


def test_versioned_pull_payload_domain_authority_remains_connected() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 4,
            "restart_epoch": 2,
            "cameras": [],
            "domains": {"fall": {"enabled": False}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-token")

    assert config.domains.fall is not None
    assert config.domains.fall.enabled is False


def test_connection_store_uses_central_edge_database() -> None:
    assert connection_store_module.EDGE_DATABASE_PATH.name == "edge.sqlite3"


def test_public_ingest_timeout_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BACKEND_INGEST_TIMEOUT_SEC", raising=False)
    assert _backend_ingest_timeout_sec() == 10.0

    monkeypatch.setenv("API_BACKEND_INGEST_TIMEOUT_SEC", "17.25")
    assert _backend_ingest_timeout_sec() == 17.25


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_public_ingest_timeout_rejects_non_positive_or_non_finite(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("API_BACKEND_INGEST_TIMEOUT_SEC", value)
    with pytest.raises(InvalidBackendIngestTimeoutError, match=value):
        _backend_ingest_timeout_sec()


def _write_fake_docker(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "${FAKE_DOCKER_LOG:?}"\n'
        'if [ "${1:-}" = compose ] && [ "${2:-}" = version ]; then exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


# Disposable base so preflight tests stay about profile overlays under the
# production contract (HTTPS Hub + unique dashboard bootstrap + RTSP flags).
# Callers still override individual keys via ``content``.
_BASE_PREFLIGHT_ENV = "\n".join(
    (
        "ML_API_IMAGE=ghcr.io/example/ml-api@sha256:test",
        "ML_WORKER_IMAGE=ghcr.io/example/ml-worker@sha256:test",
        "ML_WORKER_PROFILE=cpu-host",
        "CLIP_STORE_HOST_DIR=/tmp/clip-store-preflight",
        "API_BACKEND_BASE_URL=https://hub.example.com",
        "API_DASHBOARD_USERNAME=site-ops",
        "API_DASHBOARD_PASSWORD=disposable-bootstrap-9f3a",
        "API_EDGE_RELAY_TOKEN=disposable-relay-7c1e5b9a2f4d8e6b",
        "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS=1",
        "ML_RTSP_ALLOW_LOCAL_DESTINATIONS=0",
        "",
    )
)


def _merge_preflight_env(content: str) -> str:
    """Merge caller assignments over the disposable base (case-insensitive keys).

    Duplicate keys inside ``content`` are preserved so preflight duplicate-key
    tests still observe them; base keys are only used when content omits them.
    """

    def _parse(block: str) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            rows.append((key, key.upper(), value))
        return rows

    base_rows = _parse(_BASE_PREFLIGHT_ENV)
    content_rows = _parse(content)
    content_keys = {normalized for _, normalized, _ in content_rows}
    merged: list[str] = []
    for key, normalized, value in base_rows:
        if normalized not in content_keys:
            merged.append(f"{key}={value}")
    for key, _normalized, value in content_rows:
        merged.append(f"{key}={value}")
    return "\n".join(merged) + "\n"


def _run_preflight(
    tmp_path: Path,
    content: str,
    *compose_args: str,
) -> subprocess.CompletedProcess[str]:
    env_file = tmp_path / "edge.env"
    merged = _merge_preflight_env(content)
    env_file.write_text(merged, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    _write_fake_docker(bin_dir / "docker")
    env = {
        **os.environ,
        "FAKE_DOCKER_LOG": str(docker_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    env.pop("EDGE_RENDER_GID", None)
    env.pop("EDGE_VIDEO_GID", None)
    if re.search(r"^EDGE_RENDER_GID=", merged, re.MULTILINE):
        device = tmp_path / "renderD128"
        if not device.exists():
            device.write_bytes(b"")
        env["EDGE_RENDER_DEVICE"] = str(device)
        render_match = re.search(r"^EDGE_RENDER_GID=(\S+)", merged, re.MULTILINE)
        if render_match and re.fullmatch(r"[0-9]+", render_match.group(1)):
            env["EDGE_RENDER_DEVICE_GID"] = render_match.group(1)
    return subprocess.run(
        [
            str(ROOT / "scripts/edge-preflight/check-env.sh"),
            str(env_file),
            *compose_args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _render_invocations(tmp_path: Path) -> list[str]:
    return [
        line
        for line in (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
        if " config" in line
    ]


@pytest.mark.parametrize(
    "key",
    [
        "API_FACILITY_ID",
        "api_facility_token",
        "ML_DEFAULT_CAMERA_FPS",
        "ml_api_worker_stream_origin",
        "CLIP_STORE_DIR",
        "RELAY_URL",
    ],
)
def test_preflight_rejects_retired_alias_policy_and_internal_keys(tmp_path: Path, key: str) -> None:
    result = _run_preflight(tmp_path, f"{key}=injected\n")
    assert result.returncode != 0
    assert key in result.stderr
    assert "retired" in result.stderr


@pytest.mark.parametrize(
    ("profile", "expected_overlay"),
    [
        ("cpu", None),
        ("cpu-host", None),
        ("mps", None),
        ("apple-mps-host", None),
        ("igpu", "compose.edge.igpu.yaml"),
        ("intel-vaapi-host", "compose.edge.igpu.yaml"),
        ("cuda", "compose.edge.nvidia.yaml"),
        ("nvidia-host-bridge", "compose.edge.nvidia.yaml"),
        ("nvidia-device-experimental", "compose.edge.nvidia.yaml"),
    ],
)
def test_preflight_selects_only_the_profile_compatible_overlay(
    tmp_path: Path,
    profile: str,
    expected_overlay: str | None,
) -> None:
    extra = f"ML_WORKER_PROFILE={profile}\n"
    if expected_overlay == "compose.edge.igpu.yaml":
        extra += "EDGE_RENDER_GID=993\nEDGE_VIDEO_GID=44\n"
    result = _run_preflight(tmp_path, extra)

    assert result.returncode == 0, result.stderr
    invocations = _render_invocations(tmp_path)
    assert invocations
    assert all(
        invocation.count("compose.edge.nvidia.yaml")
        == (1 if expected_overlay == "compose.edge.nvidia.yaml" else 0)
        for invocation in invocations
    )
    assert all(
        invocation.count("compose.edge.igpu.yaml")
        == (1 if expected_overlay == "compose.edge.igpu.yaml" else 0)
        for invocation in invocations
    )


@pytest.mark.parametrize(
    "profile",
    ["cpu", "cpu-host", "mps", "apple-mps-host", "igpu", "intel-vaapi-host"],
)
def test_preflight_rejects_nvidia_overlay_for_non_nvidia_profiles(
    tmp_path: Path,
    profile: str,
) -> None:
    extra = f"ML_WORKER_PROFILE={profile}\n"
    if profile in {"igpu", "intel-vaapi-host"}:
        extra += "EDGE_RENDER_GID=993\nEDGE_VIDEO_GID=44\n"
    result = _run_preflight(
        tmp_path,
        extra,
        "-f",
        "compose.edge.nvidia.yaml",
    )

    assert result.returncode != 0
    assert profile in result.stderr
    assert "compose.edge.nvidia.yaml" in result.stderr


def test_preflight_rejects_equals_attached_nvidia_overlay_for_cpu_profile(
    tmp_path: Path,
) -> None:
    result = _run_preflight(
        tmp_path,
        "ML_WORKER_PROFILE=cpu-host\n",
        "-f=compose.edge.nvidia.yaml",
    )

    assert result.returncode != 0
    assert "cpu-host" in result.stderr
    assert "compose.edge.nvidia.yaml" in result.stderr


@pytest.mark.parametrize(
    "profile",
    ["cuda", "nvidia-host-bridge", "nvidia-device-experimental"],
)
@pytest.mark.parametrize(
    "compose_arg",
    ["--file=compose.edge.nvidia.yaml", "-f=compose.edge.nvidia.yaml"],
)
def test_preflight_accepts_explicit_nvidia_overlay_without_duplicating_it(
    tmp_path: Path,
    profile: str,
    compose_arg: str,
) -> None:
    result = _run_preflight(
        tmp_path,
        f"ML_WORKER_PROFILE={profile}\n",
        compose_arg,
    )

    assert result.returncode == 0, result.stderr
    assert all(
        invocation.count("compose.edge.nvidia.yaml") == 1
        for invocation in _render_invocations(tmp_path)
    )


def test_preflight_preserves_explicit_cpu_overlay_compatibility(tmp_path: Path) -> None:
    result = _run_preflight(
        tmp_path,
        "ML_WORKER_PROFILE=cpu\n",
        "-f",
        "compose.edge.cpu.yaml",
    )

    assert result.returncode == 0, result.stderr
    assert all(
        "compose.edge.cpu.yaml" in invocation and "compose.edge.nvidia.yaml" not in invocation
        for invocation in _render_invocations(tmp_path)
    )


def test_preflight_rejects_unknown_worker_profile_before_compose(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "ML_WORKER_PROFILE=tpu\n")

    assert result.returncode != 0
    assert "ML_WORKER_PROFILE" in result.stderr
    assert "tpu" in result.stderr
    assert "unsupported profile" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "ML_API_IMAGE=one\nML_API_IMAGE=two\n",
        "ML_API_IMAGE=one\nml_api_image=two\n",
    ],
)
def test_preflight_rejects_duplicate_keys_case_insensitively(tmp_path: Path, content: str) -> None:
    result = _run_preflight(tmp_path, content)
    assert result.returncode != 0
    assert "duplicate" in result.stderr


def test_preflight_rejects_empty_relay_token(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "API_EDGE_RELAY_TOKEN=\n")
    assert result.returncode != 0
    assert "API_EDGE_RELAY_TOKEN" in result.stderr


@pytest.mark.parametrize(
    "token",
    [
        "eldercare-internal-edge-relay",  # the shipped example value
        "<random-relay-token>",  # the .env example placeholder
        "changeme",
        "PLACEHOLDER",
        "relay-token",
        "password",
    ],
)
def test_preflight_rejects_sample_placeholder_and_weak_relay_tokens(
    tmp_path: Path, token: str
) -> None:
    result = _run_preflight(tmp_path, f"API_EDGE_RELAY_TOKEN={token}\n")
    assert result.returncode != 0
    assert "API_EDGE_RELAY_TOKEN" in result.stderr
    assert "sample/placeholder/weak" in result.stderr


def test_preflight_rejects_too_short_relay_token(tmp_path: Path) -> None:
    result = _run_preflight(tmp_path, "API_EDGE_RELAY_TOKEN=short-token\n")
    assert result.returncode != 0
    assert "API_EDGE_RELAY_TOKEN" in result.stderr
    assert "too short" in result.stderr


def test_preflight_accepts_deployment_unique_relay_token(tmp_path: Path) -> None:
    # The disposable base already carries a valid unique token; an explicit
    # high-entropy value must also pass the whole gate chain.
    result = _run_preflight(
        tmp_path, "API_EDGE_RELAY_TOKEN=b4e1c9a72f0d5836a1c7e9d2f4b60853\n"
    )
    assert result.returncode == 0, result.stderr


_SYNTHETIC_RENDER_ENV = "\n".join(
    (
        "ML_API_IMAGE=ghcr.io/example/ml-api@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ML_WORKER_IMAGE=ghcr.io/example/ml-worker@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "ML_WORKER_PROFILE=cpu-host",
        "CLIP_STORE_HOST_DIR=/tmp/seeon-edge-contract-clip-store",
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
_SYNTHETIC_LEGACY_VOLUME_NAMES = {
    "EDGE_LEGACY_CATALOG_VOLUME": "seeon-contract-legacy-catalog",
    "EDGE_LEGACY_CONNECTION_VOLUME": "seeon-contract-legacy-connection",
    "EDGE_LEGACY_WORKER_VOLUME": "seeon-contract-legacy-worker",
}
_FORBIDDEN_CUTOVER_TOKENS = (
    "happy",
    "nursing",
    "hn-",
    "COMPOSE_PROJECT_NAME",
)


def _write_render_env(path: Path, extra: str = "") -> Path:
    path.write_text(_SYNTHETIC_RENDER_ENV + extra, encoding="utf-8")
    return path


def _render_compose(
    tmp_path: Path,
    *compose_files: str,
    extra: str = "",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env_file = _write_render_env(tmp_path / "render.env", extra)
    command = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
    ]
    for compose_file in compose_files:
        command.extend(["-f", compose_file])
    command.append("config")
    rendered_env = os.environ.copy()
    rendered_env.pop("EDGE_RENDER_GID", None)
    rendered_env.pop("EDGE_VIDEO_GID", None)
    if env:
        rendered_env.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=rendered_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_rendered_config(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.returncode == 0, completed.stderr
    payload = yaml.safe_load(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def _volume_sources(service: dict[str, object]) -> list[str]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    sources: list[str] = []
    for entry in volumes:
        if isinstance(entry, dict) and entry.get("type") == "volume":
            source = entry.get("source")
            if isinstance(source, str):
                sources.append(source)
        elif isinstance(entry, str) and ":" in entry:
            sources.append(entry.split(":", 1)[0])
    return sources


def _legacy_markers_in_render(payload: dict[str, object], stdout: str) -> list[str]:
    text = stdout.lower()
    declared = payload.get("volumes", {})
    declared_names = set(declared) if isinstance(declared, dict) else set()
    return [
        marker
        for marker in _LEGACY_VOLUME_MARKERS
        if marker in declared_names or marker in text
    ]


def test_base_compose_render_is_explicit_greenfield(tmp_path: Path) -> None:
    completed = _render_compose(tmp_path, "compose.edge.yaml")
    payload = _load_rendered_config(completed)
    services = payload["services"]
    assert isinstance(services, dict)
    migrator = services["edge-db-migrator"]
    assert isinstance(migrator, dict)
    volumes = payload.get("volumes", {})
    assert isinstance(volumes, dict)

    assert migrator["command"] == [
        "python",
        "-m",
        "shared.edge_db.importer",
        "--fresh-install",
    ]
    assert set(volumes) == {"edge-state"}
    assert not _legacy_markers_in_render(payload, completed.stdout)
    assert "--require-sources" not in completed.stdout
    for service_name in ("edge-db-migrator", "ml-api", "ml-worker"):
        service = services[service_name]
        assert isinstance(service, dict)
        assert "edge-state" in _volume_sources(service)
        assert not any(
            marker in source
            for source in _volume_sources(service)
            for marker in _LEGACY_VOLUME_MARKERS
        )
    for token in _FORBIDDEN_CUTOVER_TOKENS:
        assert token not in completed.stdout.lower()


def test_cutover_overlay_render_binds_three_external_legacy_volumes(
    tmp_path: Path,
) -> None:
    extra = "".join(
        f"{key}={value}\n" for key, value in _SYNTHETIC_LEGACY_VOLUME_NAMES.items()
    )
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.migrate.yaml",
        extra=extra,
        env={"COMPOSE_PROJECT_NAME": "stale-compose-project"},
    )
    payload = _load_rendered_config(completed)
    services = payload["services"]
    assert isinstance(services, dict)
    migrator = services["edge-db-migrator"]
    assert isinstance(migrator, dict)
    volumes = payload.get("volumes", {})
    assert isinstance(volumes, dict)

    assert migrator["command"] == [
        "python",
        "-m",
        "shared.edge_db.importer",
        "--require-sources",
        "catalog,connection,worker",
        "--catalog",
        "/var/lib/legacy-catalog/catalog.sqlite3",
        "--connection",
        "/var/lib/legacy-connection/connection-settings.sqlite3",
        "--worker",
        "/var/lib/legacy-worker/worker-state.sqlite3",
    ]
    assert set(volumes) >= {
        "edge-state",
        "legacy-catalog",
        "legacy-connection",
        "legacy-worker",
    }
    assert not _legacy_markers_in_render(payload, completed.stdout)
    for logical_name, expected_name in (
        ("legacy-catalog", "seeon-contract-legacy-catalog"),
        ("legacy-connection", "seeon-contract-legacy-connection"),
        ("legacy-worker", "seeon-contract-legacy-worker"),
    ):
        declaration = volumes[logical_name]
        assert isinstance(declaration, dict)
        assert declaration["name"] == expected_name
        assert declaration["external"] is True

    migrator_volumes = migrator["volumes"]
    assert isinstance(migrator_volumes, list)
    by_source = {
        entry["source"]: entry
        for entry in migrator_volumes
        if isinstance(entry, dict) and entry.get("type") == "volume"
    }
    assert by_source["legacy-catalog"]["target"] == "/var/lib/legacy-catalog"
    assert by_source["legacy-connection"]["target"] == "/var/lib/legacy-connection"
    assert by_source["legacy-worker"]["target"] == "/var/lib/legacy-worker"
    legacy_targets = {
        by_source[name]["target"]
        for name in ("legacy-catalog", "legacy-connection", "legacy-worker")
    }
    assert len(legacy_targets) == 3
    for source in ("legacy-catalog", "legacy-connection", "legacy-worker"):
        assert by_source[source]["read_only"] is True
    for runtime_name in ("ml-api", "ml-worker"):
        runtime = services[runtime_name]
        assert isinstance(runtime, dict)
        assert not any(
            source.startswith("legacy-") for source in _volume_sources(runtime)
        )
    assert "stale-compose-project" not in {
        declaration.get("name")
        for declaration in volumes.values()
        if isinstance(declaration, dict)
    }
    for token in _FORBIDDEN_CUTOVER_TOKENS:
        assert token not in completed.stdout.lower()
    assert "--fresh-install" not in completed.stdout


@pytest.mark.parametrize("missing", sorted(LEGACY_VOLUME_BINDINGS))
def test_cutover_overlay_rejects_missing_legacy_volume_binding_before_docker(
    tmp_path: Path,
    missing: str,
) -> None:
    extra = "".join(
        f"{key}={value}\n"
        for key, value in _SYNTHETIC_LEGACY_VOLUME_NAMES.items()
        if key != missing
    )
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.migrate.yaml",
        extra=extra,
    )

    assert completed.returncode != 0
    assert missing in completed.stderr
    assert "required variable" in completed.stderr
    assert completed.stdout.strip() == ""


@pytest.mark.parametrize(
    "extra",
    [
        "EDGE_LEGACY_CATALOG_VOLUME=\n"
        "EDGE_LEGACY_CONNECTION_VOLUME=seeon-contract-legacy-connection\n"
        "EDGE_LEGACY_WORKER_VOLUME=seeon-contract-legacy-worker\n",
        "EDGE_LEGACY_CATALOG_VOLUME=seeon-contract-legacy-catalog\n"
        "EDGE_LEGACY_CONNECTION_VOLUME=   \n"
        "EDGE_LEGACY_WORKER_VOLUME=seeon-contract-legacy-worker\n",
        "EDGE_LEGACY_CATALOG_VOLUME=seeon-contract-legacy-catalog\n"
        "EDGE_LEGACY_CATALOG_VOLUME=\n"
        "EDGE_LEGACY_CONNECTION_VOLUME=seeon-contract-legacy-connection\n"
        "EDGE_LEGACY_WORKER_VOLUME=seeon-contract-legacy-worker\n",
    ],
)
def test_cutover_overlay_rejects_empty_or_blank_legacy_volume_bindings(
    tmp_path: Path,
    extra: str,
) -> None:
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.migrate.yaml",
        extra=extra,
    )

    assert completed.returncode != 0
    assert "required variable" in completed.stderr
    assert completed.stdout.strip() == ""


def test_example_env_documents_cutover_bindings_without_assigning_them() -> None:
    example = (ROOT / ".env.edge.prod.example").read_text(encoding="utf-8")
    assigned = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", example, re.MULTILINE)
    )
    commented = set(
        re.findall(r"^#\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE)
    )

    assert MIGRATION_OVERLAY_FILE.name in example
    assert LEGACY_VOLUME_BINDINGS.isdisjoint(assigned)
    assert LEGACY_VOLUME_BINDINGS <= commented
    for token in ("happy", "nursing", "hn-", "COMPOSE_PROJECT_NAME"):
        assert token not in example.lower()


def _worker_group_add(payload: dict[str, object]) -> list[str]:
    services = payload["services"]
    assert isinstance(services, dict)
    worker = services["ml-worker"]
    assert isinstance(worker, dict)
    group_add = worker.get("group_add", [])
    assert isinstance(group_add, list)
    return [str(value) for value in group_add]


def test_igpu_overlay_does_not_hardcode_host_gids_or_docker_privilege(
    tmp_path: Path,
) -> None:
    overlay = (ROOT / "compose.edge.igpu.yaml").read_text(encoding="utf-8")
    assert "EDGE_RENDER_GID" in overlay
    assert "EDGE_VIDEO_GID" in overlay
    assert '"104"' not in overlay
    assert "- \"104\"" not in overlay
    for token in ("docker.sock", "privileged", "usermod", "-G docker"):
        assert token not in overlay
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.igpu.yaml",
        extra=(
            "ML_WORKER_PROFILE=intel-vaapi-host\n"
            "EDGE_RENDER_GID=993\n"
            "EDGE_VIDEO_GID=44\n"
        ),
    )
    payload = _load_rendered_config(completed)
    assert _worker_group_add(payload) == ["993", "44"]
    for token in ("docker.sock", "privileged", "usermod", "-G docker"):
        assert token not in completed.stdout


IGPU_GID_BINDINGS = {
    "EDGE_RENDER_GID": "993",
    "EDGE_VIDEO_GID": "44",
}


def test_igpu_overlay_renders_injected_gids_exactly(tmp_path: Path) -> None:
    extra = (
        "ML_WORKER_PROFILE=intel-vaapi-host\n"
        + "".join(f"{key}={value}\n" for key, value in IGPU_GID_BINDINGS.items())
    )
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.igpu.yaml",
        extra=extra,
        env={"COMPOSE_PROJECT_NAME": "stale-compose-project"},
    )
    payload = _load_rendered_config(completed)
    assert _worker_group_add(payload) == ["993", "44"]
    assert completed.stdout.count("993") >= 1
    assert "group_add" in completed.stdout
    for stale in ("104", "1", "2"):
        assert stale not in _worker_group_add(payload)
    for token in ("docker.sock", "privileged", "usermod"):
        assert token not in completed.stdout


@pytest.mark.parametrize("missing", sorted(IGPU_GID_BINDINGS))
def test_igpu_overlay_rejects_missing_gid_before_docker(
    tmp_path: Path, missing: str
) -> None:
    extra = (
        "ML_WORKER_PROFILE=intel-vaapi-host\n"
        + "".join(
            f"{key}={value}\n"
            for key, value in IGPU_GID_BINDINGS.items()
            if key != missing
        )
    )
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.igpu.yaml",
        extra=extra,
    )
    assert completed.returncode != 0
    assert missing in completed.stderr
    assert "required variable" in completed.stderr
    assert completed.stdout.strip() == ""


@pytest.mark.parametrize(
    "extra",
    [
        "ML_WORKER_PROFILE=intel-vaapi-host\nEDGE_RENDER_GID=\nEDGE_VIDEO_GID=44\n",
        "ML_WORKER_PROFILE=intel-vaapi-host\nEDGE_RENDER_GID=993\nEDGE_VIDEO_GID=   \n",
        "ML_WORKER_PROFILE=intel-vaapi-host\nEDGE_RENDER_GID=993\nEDGE_RENDER_GID=\nEDGE_VIDEO_GID=44\n",
    ],
)
def test_igpu_overlay_rejects_empty_or_blank_gids(tmp_path: Path, extra: str) -> None:
    completed = _render_compose(
        tmp_path,
        "compose.edge.yaml",
        "compose.edge.igpu.yaml",
        extra=extra,
    )
    assert completed.returncode != 0
    assert "required variable" in completed.stderr
    assert completed.stdout.strip() == ""


def test_example_env_documents_gid_bindings_without_assigning_them() -> None:
    example = (ROOT / ".env.edge.prod.example").read_text(encoding="utf-8")
    assigned = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    commented = set(re.findall(r"^#\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
    assert set(IGPU_GID_BINDINGS).isdisjoint(assigned)
    assert set(IGPU_GID_BINDINGS) <= commented
    assert "104" not in example
    video_gid_tail = example.split("EDGE_VIDEO_GID=", 1)[-1][:20]
    assert "EDGE_VIDEO_GID=" in example
    assert "44" not in video_gid_tail
