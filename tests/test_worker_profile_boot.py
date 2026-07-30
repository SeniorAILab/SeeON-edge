from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

from edge.runtime.profile import BootDependencies, VerifyResult, profile_verify_stage


@pytest.mark.parametrize(
    ("profile", "legacy_env", "device_ok", "decode_ok"),
    [
        (None, {}, True, True),
        ("unknown", {}, True, True),
        ("cpu", {}, False, True),
        ("cpu", {}, True, False),
        ("cpu", {"ML_RTSP_BACKEND": "nvdec"}, True, True),
        ("cpu", {"ML_DEFAULT_DECODE_BACKEND": "nvdec"}, True, True),
    ],
    ids=(
        "missing-profile",
        "unknown-profile",
        "device-verify-failure",
        "decode-preflight-failure",
        "rtsp-backend-conflict",
        "default-decode-backend-conflict",
    ),
)
def test_main_refuses_invalid_profile_before_config_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    profile: str | None,
    legacy_env: dict[str, str],
    device_ok: bool,
    decode_ok: bool,
) -> None:
    worker = importlib.import_module("edge.runtime.edge_worker")
    for name in ("ML_WORKER_PROFILE", "ML_RTSP_BACKEND", "ML_DEFAULT_DECODE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    if profile is not None:
        monkeypatch.setenv("ML_WORKER_PROFILE", profile)
    for name, value in legacy_env.items():
        monkeypatch.setenv(name, value)

    dependencies = BootDependencies(
        {
            "cpu": lambda: VerifyResult(
                device_ok, "cpu", "device", "injected device result"
            )
        }
    )

    def decode_probe(_: str) -> VerifyResult:
        return VerifyResult(decode_ok, "cpu", "decode", "injected decode result")

    monkeypatch.setattr(
        worker,
        "profile_verify_stage",
        lambda env: profile_verify_stage(env, dependencies, decode_probe),
    )

    def config_must_not_load(_: object) -> object:
        raise AssertionError("config acquisition ran after failed profile verification")

    monkeypatch.setattr(worker, "_load_startup_config", config_must_not_load)

    assert worker.main([]) == 3


def test_check_config_does_not_require_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(
            edge_config_payload(
                camera_count=1, include_optional_fields=False, resident_ids=False
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ML_WORKER_PROFILE", raising=False)

    worker = importlib.import_module("edge.runtime.edge_worker")

    assert worker.main(["--config", str(config_path), "--check-config"]) == 0
