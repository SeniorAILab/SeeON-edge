from __future__ import annotations

from pathlib import Path

from worker.runtime.config.lkg_store import (
    CONFIG_HISTORY_RETENTION_COUNT,
    WorkerConfigLkgStore,
)
from worker.runtime.config.restart import RestartDirective


def _config_payload(config_version: int) -> dict[str, object]:
    return {
        "registry_version": config_version,
        "config_version": config_version,
        "restart_epoch": 1,
        "cameras": [],
    }


def test_config_revision_is_durable_and_contains_the_resolved_payload(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=7), RestartDirective(generation=1, version=7, registry=7)
    )
    revisions = tuple((config_store.database_path / "revisions").glob("*.json"))
    assert len(revisions) == 1
    assert '"config_version":7' in revisions[0].read_text()


def test_config_revisions_are_bounded_beyond_retention_window(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1, registry=1)
    )

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version, registry=version),
        )

    remaining = _revision_versions(config_store)
    assert 1 not in remaining
    assert len(remaining) == CONFIG_HISTORY_RETENTION_COUNT


def test_config_revisions_keep_the_newest_versions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1, registry=1)
    )

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version, registry=version),
        )

    remaining = _revision_versions(config_store)
    assert 1 not in remaining
    assert len(remaining) == CONFIG_HISTORY_RETENTION_COUNT


def _revision_versions(store: WorkerConfigLkgStore) -> set[int]:
    return {
        int(path.read_text().split('"config_version":', 1)[1].split(",", 1)[0])
        for path in (store.database_path / "revisions").glob("*.json")
    }


def test_worker_and_compose_have_no_facility_identity_env_contract() -> None:
    root = Path(__file__).resolve().parents[1]

    assert "API_FACILITY_ID_ENV" not in (root / "worker/runtime/config/config_pull.py").read_text()
    compose = (root / "compose.edge.yaml").read_text()
    assert "API_FACILITY_ID:" not in compose
    assert "EDGE_FACILITY_TOKEN:" not in compose
