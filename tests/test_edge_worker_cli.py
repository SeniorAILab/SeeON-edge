from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

import edge.runtime.edge_worker as edge_worker
from edge.runtime.edge_worker import main


def test_edge_worker_cli_check_config(tmp_path: Path) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(edge_config_payload(camera_count=1, include_optional_fields=False)),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--check-config"]) == 0


def test_edge_worker_cli_rejects_non_positive_max_frames() -> None:
    assert main(["--max-frames-per-camera", "0"]) == 2


def test_edge_worker_cli_requires_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)

    assert main(["--check-config"]) == 2


def test_edge_worker_cli_rejects_unknown_arguments() -> None:
    assert main(["--unknown-option"]) == 2


def test_edge_worker_loads_yaml_when_config_env_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(edge_config_payload(camera_count=1, include_optional_fields=False)),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGE_CAMERA_CONFIG", str(config_path))
    monkeypatch.delenv("RELAY_URL", raising=False)

    startup = edge_worker._load_startup_config(edge_worker._parse_args([]))

    assert startup.source == "yaml"
    assert startup.pulled is None
    assert startup.config.cameras[0].camera_id == "camera-1"


def test_edge_worker_pulls_config_when_config_env_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(edge_config_payload(camera_count=1, include_optional_fields=False)),
        encoding="utf-8",
    )
    pulled_config = edge_worker.load_edge_worker_config(config_path)
    calls: list[tuple[str, str | None]] = []

    def fake_pull(relay_url: str, relay_token: str | None):
        calls.append((relay_url, relay_token))
        return pulled_config, 7, "relay"

    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(edge_worker, "load_edge_worker_config_from_relay", fake_pull)

    startup = edge_worker._load_startup_config(edge_worker._parse_args([]))

    assert calls == [("http://ml-api:8000", "relay-token")]
    assert startup.source == "relay"
    assert startup.registry_version == 7
    assert startup.yaml_config is None
