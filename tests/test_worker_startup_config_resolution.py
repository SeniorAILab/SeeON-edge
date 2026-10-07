from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import worker.__main__ as worker_main
from worker.runtime.config import (
    CameraRuntimeConfig,
    ConfigSnapshot,
    ConfigSource,
    RelayConfig,
    RestartDirective,
    WorkerConfig,
    WorkerConfigError,
    WorkerConfigLkgStore,
)
from worker.runtime.worker import WorkerRuntime


@pytest.fixture(autouse=True)
def _no_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("RELAY_TOKEN", raising=False)


def _write_yaml_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "relay": {"url": "http://ml-api:8000", "token": "relay-token"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _worker_config(camera_id: str) -> WorkerConfig:
    return WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="relay-token"),
        cameras=(
            CameraRuntimeConfig(
                camera_id=camera_id,
                facility_id="facility-1",
                rtsp_url=f"rtsp://{camera_id}/sub",
            ),
        ),
    )


def _spy_workerruntime_config(monkeypatch: pytest.MonkeyPatch) -> list[WorkerRuntime]:
    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)
    return constructed


def test_no_yaml_successful_pull_becomes_live_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    pulled_config = _worker_config("pulled-camera")
    snapshot = ConfigSnapshot(
        config=pulled_config,
        registry_version=4,
        directive=RestartDirective(generation=1, version=4),
        source=ConfigSource.PULLED,
        stale=False,
    )

    def _fake_load_from_relay(
        relay_url: str, relay_token: str | None, **_kwargs: object
    ) -> ConfigSnapshot:
        assert relay_url == "http://ml-api:8000"
        assert relay_token == "relay-token"
        return snapshot

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fake_load_from_relay)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is pulled_config


def test_no_yaml_failed_pull_with_lkg_uses_lkg_config_and_logs_stale(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    lkg_config = _worker_config("lkg-camera")
    snapshot = ConfigSnapshot(
        config=lkg_config,
        registry_version=2,
        directive=RestartDirective(generation=1, version=2),
        source=ConfigSource.LKG,
        stale=True,
    )
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: snapshot)
    constructed = _spy_workerruntime_config(monkeypatch)

    with caplog.at_level(logging.INFO):
        exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is lkg_config
    assert any(
        "lkg" in record.getMessage().lower() and "stale=True" in record.getMessage()
        for record in caplog.records
    )


def test_no_yaml_failed_pull_no_lkg_exits_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "http://ml-api:8000" in message and "RELAY_TOKEN" in message and "dashboard" in message
        for message in messages
    )


def test_check_config_no_yaml_missing_relay_token_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check-config must not pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main(["--check-config"]) == 2


def test_yaml_set_successful_pull_wins_over_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)
    pulled_config = _worker_config("pulled-camera")
    snapshot = ConfigSnapshot(
        config=pulled_config,
        registry_version=9,
        directive=RestartDirective(generation=2, version=9),
        source=ConfigSource.PULLED,
        stale=False,
    )
    monkeypatch.setattr(worker_main, "resolve_startup_config", lambda *_a, **_k: snapshot)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is pulled_config


def test_yaml_set_failed_pull_no_lkg_falls_back_to_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)
    captured_yaml_config: list[WorkerConfig] = []

    def _fake_resolve_startup_config(
        yaml_config: WorkerConfig, relay_url: str, relay_token: str | None
    ) -> ConfigSnapshot:
        captured_yaml_config.append(yaml_config)
        return ConfigSnapshot(
            config=yaml_config,
            registry_version=0,
            directive=RestartDirective(generation=0, version=0),
            source=ConfigSource.YAML,
            stale=True,
        )

    monkeypatch.setattr(worker_main, "resolve_startup_config", _fake_resolve_startup_config)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    assert len(captured_yaml_config) == 1
    assert constructed[0].config is captured_yaml_config[0]


def test_check_config_no_yaml_is_strictly_static_no_pull_no_lkg_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must never attempt a relay pull")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    def _fail_save(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must never write the LKG store")

    monkeypatch.setattr(WorkerConfigLkgStore, "save", _fail_save)

    def _fail_construct(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must not construct WorkerRuntime")

    monkeypatch.setattr(worker_main, "make_restart_check", _fail_construct)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail_construct)

    assert worker_main.main(["--check-config"]) == 0


def test_check_config_no_yaml_reports_lkg_presence_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "resolve_state_dir", lambda: tmp_path)

    with caplog.at_level(logging.INFO):
        exit_code = worker_main.main(["--check-config"])

    assert exit_code == 0
    assert any("no last-known-good cache yet" in record.getMessage() for record in caplog.records)


def test_no_yaml_missing_relay_token_exits_before_any_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELAY_TOKEN", raising=False)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


def test_normal_runtime_rejects_retired_relay_url_before_any_pull(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://other-relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not pull when retired RELAY_URL is present")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    with caplog.at_level(logging.ERROR):
        assert worker_main.main([]) == 2

    assert any(
        "RELAY_URL" in record.getMessage()
        and "versioned worker config authority" in record.getMessage()
        for record in caplog.records
    )


def test_no_yaml_pull_raising_workerconfigerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        raise WorkerConfigError("worker config LKG revision mismatch")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _raise)

    assert worker_main.main([]) == 2


def test_yaml_set_pull_raising_workerconfigerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        raise WorkerConfigError("worker config LKG revision mismatch")

    monkeypatch.setattr(worker_main, "resolve_startup_config", _raise)

    assert worker_main.main(["--config", str(config_path)]) == 2


def test_no_yaml_pull_raising_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _raise)

    assert worker_main.main([]) == 2


def test_yaml_set_pull_raising_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "resolve_startup_config", _raise)

    assert worker_main.main(["--config", str(config_path)]) == 2


def test_no_yaml_missing_packaged_model_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise WorkerConfigError(
            "packaged default pose+bbox56 fall model is not fully provisioned at "
            "'models/fall/pose-bbox56-gru'; run scripts/fetch-models.sh"
        )

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _raise)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "a relay pull must not be attempted once local model resolution has failed"
        )

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    assert any("model" in record.getMessage().lower() for record in caplog.records)


def test_no_yaml_local_override_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> object:
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _raise)
    monkeypatch.setattr(
        worker_main,
        "load_worker_config_from_relay",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("must not pull once local model resolution has failed")
        ),
    )

    assert worker_main.main([]) == 2


def test_check_config_no_yaml_never_resolves_local_model_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "--check-config must not resolve local model overrides -- it is a "
            "static, no-side-effect check that must pass without a provisioned model"
        )

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _fail)

    assert worker_main.main(["--check-config"]) == 0


def test_no_yaml_no_config_message_names_both_plausible_causes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("unreachable" in message and "no usable camera" in message for message in messages)


def test_check_config_with_yaml_never_calls_resolve_startup_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "--check-config with an explicit YAML must validate the YAML alone, "
            "matching the pre-#27 no-relay-side-effect contract"
        )

    monkeypatch.setattr(worker_main, "resolve_startup_config", _fail)
    monkeypatch.setattr(worker_main, "make_restart_check", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    assert worker_main.main(["--config", str(config_path), "--check-config"]) == 0


def test_worker_config_accepts_zero_cameras() -> None:
    config = WorkerConfig(relay=RelayConfig(url="http://ml-api:8000", token="relay-token"))

    assert config.cameras == ()


def test_worker_config_still_rejects_a_config_missing_the_relay_section() -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"cameras": []})


def test_no_yaml_pull_resolving_to_zero_cameras_still_boots(
    monkeypatch: pytest.MonkeyPatch,
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    zero_camera_config = WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="relay-token"),
    )
    snapshot = ConfigSnapshot(
        config=zero_camera_config,
        registry_version=1,
        directive=RestartDirective(generation=0, version=1),
        source=ConfigSource.PULLED,
        stale=False,
    )
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: snapshot)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config.cameras == ()


def test_no_yaml_and_no_relay_token_still_exits_fast_with_zero_cameras_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


def test_packaged_fall_config_requires_the_56_wide_pose_bbox_contract(
    packaged_fall_bundle: Path,
) -> None:
    from pydantic import ValidationError

    from worker.runtime.config.worker_models import FallModelConfig

    def _config(width: int) -> FallModelConfig:
        return FallModelConfig(
            type="pose-bbox56-proxy-v0",
            framework="pytorch",
            mode="sequence",
            artifact_dir=packaged_fall_bundle,
            window=30,
            stride=5,
            input_shape=(30, width),
            operating_threshold=0.5,
        )

    accepted = _config(56)
    assert accepted.input_shape == (30, 56)
    assert accepted.framework == "pytorch"

    with pytest.raises(ValidationError, match=r"input_shape must be \[window, 56\]"):
        _config(51)
