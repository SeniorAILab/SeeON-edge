import ipaddress
import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def packaged_fall_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
    from worker.runtime.config import local_env

    artifact = write_pose_bbox56_bundle(tmp_path / "models" / "fall" / "pose-bbox56-gru")
    monkeypatch.setattr(local_env, "_DEFAULT_ARTIFACT_DIR", str(artifact))
    return artifact


@pytest.fixture(autouse=True)
def explicit_dashboard_bootstrap_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setenv("API_DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("API_DASHBOARD_PASSWORD", "admin")
    yield


@pytest.fixture(autouse=True)
def allow_insecure_hub_http_for_local_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setenv("API_BACKEND_ALLOW_INSECURE_HTTP", "1")
    yield


@pytest.fixture(autouse=True)
def isolate_state_dir_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    yield


@pytest.fixture(autouse=True)
def stub_rtsp_hostname_resolution(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from shared import rtsp_url_policy

    def _stub(hostname: str) -> tuple[str, ...]:
        host = hostname.strip().lower().rstrip(".")
        if not host:
            return ()
        try:
            return (str(ipaddress.ip_address(host)),)
        except ValueError:
            if host == "localhost" or host.endswith(".localhost"):
                return ("127.0.0.1",)
            return ("8.8.8.8",)

    monkeypatch.setattr(rtsp_url_policy, "resolve_host_a_aaaa", _stub)
    yield


@pytest.fixture(autouse=True)
def _deterministic_file_modes() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


pytest_plugins = ("tests_support.private_bundle",)
