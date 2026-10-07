from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from worker.pipeline.output.evidence.clip_config import DEFAULT_CLIP_STORE_DIR
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
from worker.runtime.config.worker_models import ClipRecordingConfig
from worker.runtime.worker import WorkerRuntime


def _camera_payload(camera_id: str = "camera-1") -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "facility_id": "facility-1",
        "rtsp_url": "rtsp://camera/stream",
    }


def test_resolved_domain_enabled_parses_a_well_formed_map() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "domains": {"fall": {"enabled": True}, "bed_exit": {"enabled": False}},
        }
    )
    assert payload.resolved_domain_enabled == {"fall": True, "bed_exit": False}


def test_resolved_domain_enabled_is_empty_when_domains_absent() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    assert payload.resolved_domain_enabled == {}
    assert payload.domains is None


@pytest.mark.parametrize(
    "domains_value",
    [
        {"fall": "not-an-object"},
        {"fall": {"enabled": "not-a-bool"}},
        {"unknown-domain": {"enabled": True}},
        {"": {"enabled": True}},
    ],
)
def test_resolved_domain_enabled_drops_only_the_malformed_entry_and_logs(
    domains_value: dict[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "domains": {**domains_value, "bed_exit": {"enabled": True}},
        }
    )

    resolved = payload.resolved_domain_enabled

    assert resolved == {"bed_exit": True}
    assert capsys.readouterr().err


def test_resolved_clip_store_subdir_accepts_a_relative_path() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": "backup-drive/clips",
        }
    )
    assert payload.resolved_clip_store_subdir == "backup-drive/clips"


def test_resolved_clip_store_subdir_is_none_when_absent() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    assert payload.resolved_clip_store_subdir is None


@pytest.mark.parametrize(
    "value",
    ["/etc/passwd", "../escape", "sub/../../escape", "", "   ", 123, {"path": "sub"}],
)
def test_resolved_clip_store_subdir_falls_open_to_none_on_malformed_value(
    value: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": value,
        }
    )

    assert payload.resolved_clip_store_subdir is None
    assert capsys.readouterr().err


def test_to_worker_config_with_no_domains_override_preserves_camera_domains_default() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.enabled == ("fall",)
    assert config.clip.store_subdir is None


def test_to_worker_config_domains_override_replaces_camera_domains_entirely() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
            "domains": {"fall": {"enabled": False}, "bed_exit": {"enabled": True}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.resolved_overrides() == {"fall": False, "bed_exit": True}
    assert config.enabled_domains == ("bed_exit",)


def test_to_worker_config_domains_override_can_represent_all_domains_off() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall", "bed_exit"]}],
            "domains": {"fall": {"enabled": False}, "bed_exit": {"enabled": False}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.resolved_overrides() == {"fall": False, "bed_exit": False}
    assert config.enabled_domains == ()


def test_to_worker_config_domains_override_is_a_partial_overlay_not_a_replace() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "domains": {"fall": {"enabled": False}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.resolved_overrides() == {"fall": False}
    assert config.enabled_domains == ("bed_exit",)


def test_to_worker_config_with_no_domains_signal_resolves_to_registry_defaults() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.resolved_overrides() == {}
    assert config.enabled_domains == ("fall", "bed_exit")


def test_to_worker_config_with_explicit_empty_camera_domains_list_stays_off() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": []}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.enabled_domains == ()


def test_to_worker_config_with_specific_camera_domains_resolves_exactly_as_given() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.enabled_domains == ("fall",)


def test_to_worker_config_clip_store_subdir_merges_into_a_local_clip_config() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": "external-drive",
        }
    )
    local_clip = ClipRecordingConfig(enabled=True)

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret", clip=local_clip)

    assert config.clip.enabled is True
    assert config.clip.store_subdir == "external-drive"


def test_to_worker_config_without_clip_store_subdir_leaves_local_clip_config_untouched() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    local_clip = ClipRecordingConfig(enabled=True, store_subdir="already-set")

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret", clip=local_clip)

    assert config.clip.store_subdir == "already-set"


def test_clip_recording_config_accepts_a_relative_store_subdir() -> None:
    config = ClipRecordingConfig(store_subdir="sub/dir")
    assert config.store_subdir == "sub/dir"


@pytest.mark.parametrize("value", ["/absolute", "sub/../escape", ".."])
def test_clip_recording_config_rejects_unsafe_store_subdir(value: str) -> None:
    with pytest.raises(ValidationError):
        ClipRecordingConfig(store_subdir=value)


def _fake_runtime(
    store_subdir: str | None,
    *,
    clip_store_dir: Path = Path(DEFAULT_CLIP_STORE_DIR),
) -> WorkerRuntime:
    from types import SimpleNamespace

    return SimpleNamespace(
        config=SimpleNamespace(clip=SimpleNamespace(store_subdir=store_subdir)),
        _clip_store_dir=clip_store_dir,
    )


def test_resolved_clip_store_dir_is_the_bare_root_when_no_subdir_selected() -> None:
    runtime = _fake_runtime(None)

    resolved = WorkerRuntime._resolved_clip_store_dir(runtime)

    assert resolved == Path(DEFAULT_CLIP_STORE_DIR)


def test_resolved_clip_store_dir_appends_a_selected_subdir(tmp_path: Path) -> None:
    runtime = _fake_runtime("backup/clips", clip_store_dir=tmp_path)

    resolved = WorkerRuntime._resolved_clip_store_dir(runtime)

    assert resolved == tmp_path / "backup" / "clips"


@pytest.mark.parametrize("unsafe_subdir", ["/etc/passwd", "../escape", "sub/../../escape"])
def test_resolved_clip_store_dir_defensively_rejects_an_unsafe_subdir(
    tmp_path: Path, unsafe_subdir: str
) -> None:
    runtime = _fake_runtime(unsafe_subdir, clip_store_dir=tmp_path)

    with pytest.raises(RuntimeError, match="relative and traversal-free"):
        WorkerRuntime._resolved_clip_store_dir(runtime)
