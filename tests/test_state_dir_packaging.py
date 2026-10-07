from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from backend.app.edge_db import EDGE_DATABASE_PATH, EDGE_STATE_DIRECTORY
from worker.runtime.config.lkg_store import WorkerConfigLkgStore
from worker.runtime.lease import GPU_LEASE_FILENAME

ROOT = Path(__file__).resolve().parents[1]


class _ComposeLoader(yaml.SafeLoader):
    ...


_ComposeLoader.add_constructor("!reset", lambda loader, node: None)

EXPECTED_WORKER_STATE_DIR = "/root/.local/state/ml-worker"
EXPECTED_API_STATE_DIR = "/root/.local/state/ml-api"
EXPECTED_EDGE_STATE_DIR = "/var/lib/seeon-state"
EXPECTED_EDGE_DATABASE = "/var/lib/seeon-state/edge.sqlite3"

WORKER_RESOLVER_PATH = ROOT / "worker" / "runtime" / "state_dir.py"
API_RESOLVER_PATH = ROOT / "backend" / "app" / "shared" / "state_dir.py"


def _dockerfile_mkdir_p_args(dockerfile_name: str) -> list[str]:
    text = (ROOT / dockerfile_name).read_text(encoding="utf-8")
    args: list[str] = []
    instructions: list[str] = []
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        current = f"{current} {stripped}".strip()
        if stripped.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        instructions.append(current)
        current = ""
    if current:
        instructions.append(current)

    for instruction in instructions:
        marker = "mkdir -p "
        idx = instruction.find(marker)
        if idx == -1:
            continue
        args.extend(instruction[idx + len(marker) :].split())
    return args


def _compose_named_volume_target(compose: dict, service: str, volume_name: str) -> str:
    entries = compose["services"][service]["volumes"]
    for entry in entries:
        if isinstance(entry, str) and entry.startswith(f"{volume_name}:"):
            _, target = entry.split(":", 1)
            return target
    raise AssertionError(f"no {volume_name!r} volume mounted on service {service!r}")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.load(
        (ROOT / "compose.edge.yaml").read_text(encoding="utf-8"), Loader=_ComposeLoader
    )


def test_dockerfile_edge_mkdirs_worker_state_dir() -> None:
    args = _dockerfile_mkdir_p_args("Dockerfile.edge")
    assert EXPECTED_WORKER_STATE_DIR in args, (
        f"Dockerfile.edge must `RUN mkdir -p {EXPECTED_WORKER_STATE_DIR}`; "
        f"found mkdir -p args: {args}"
    )


def test_dockerfile_backend_mkdirs_api_state_dir() -> None:
    args = _dockerfile_mkdir_p_args("Dockerfile.backend")
    assert EXPECTED_API_STATE_DIR in args, (
        f"Dockerfile.backend must `RUN mkdir -p {EXPECTED_API_STATE_DIR}`; "
        f"found mkdir -p args: {args}"
    )


def test_dockerfiles_declare_no_volume_for_state_dir() -> None:
    volume_instruction = re.compile(r"^\s*VOLUME\b", re.MULTILINE)
    for name in ("Dockerfile.edge", "Dockerfile.backend"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not volume_instruction.search(text), (
            f"{name} must not declare a VOLUME instruction for the image-owned "
            "state dir (prevents anonymous-volume sprawl and derived-image RUN "
            "neutralization)"
        )


@pytest.mark.parametrize("service", ["edge-db-cutover", "ml-api"])
def test_compose_mounts_central_state_at_baked_path(compose: dict, service: str) -> None:
    target = _compose_named_volume_target(compose, service, "edge-state")
    assert target == EXPECTED_EDGE_STATE_DIR


def test_compose_no_longer_sets_ml_worker_state_dir_env() -> None:
    text = (ROOT / "compose.edge.yaml").read_text(encoding="utf-8")
    assert "ML_WORKER_STATE_DIR" not in text, (
        "ML_WORKER_STATE_DIR must be removed from compose.edge.yaml — production "
        "path ownership belongs to the Docker image, not an env override"
    )


def test_worker_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = worker_resolve_state_dir("ml-worker")

    assert str(resolved) == EXPECTED_WORKER_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.edge")
    assert _compose_named_volume_target(compose, "ml-worker", "worker-local-state") == (
        EXPECTED_EDGE_STATE_DIR
    )


def test_api_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from backend.app.shared.state_dir import resolve_state_dir as api_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = api_resolve_state_dir("ml-api")

    assert str(resolved) == EXPECTED_API_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.backend")
    assert _compose_named_volume_target(compose, "ml-api", "edge-state") != str(resolved)


def test_worker_resolver_reads_no_environment_override() -> None:
    source = WORKER_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "worker/runtime/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


def test_api_resolver_reads_no_environment_override() -> None:
    source = API_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "backend/app/shared/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


def test_shared_edge_database_path_matches_compose_mount(compose: dict) -> None:
    assert str(EDGE_STATE_DIRECTORY) == EXPECTED_EDGE_STATE_DIR
    assert str(EDGE_DATABASE_PATH) == EXPECTED_EDGE_DATABASE
    assert _compose_named_volume_target(compose, "ml-api", "edge-state") == str(
        EDGE_DATABASE_PATH.parent
    )
    assert not any(
        str(volume).startswith("edge-state:")
        for volume in compose["services"]["ml-worker"]["volumes"]
    )


def test_worker_config_cache_is_local_to_the_worker_state_directory() -> None:
    worker_path = WorkerConfigLkgStore().database_path

    assert worker_path.name == "config-lkg"
    assert worker_path.parent == Path.home() / ".local/state/ml-worker"


def test_gpu_lease_uses_worker_local_state_not_central_database_volume(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    lease_path = worker_resolve_state_dir("ml-worker") / GPU_LEASE_FILENAME

    assert str(lease_path.parent) == EXPECTED_WORKER_STATE_DIR
    assert _compose_named_volume_target(compose, "ml-worker", "worker-local-state") != str(
        lease_path.parent
    )
