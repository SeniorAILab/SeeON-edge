from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

from edge.runtime.profile.registry import PROFILE_REGISTRY


def test_worker_entrypoint_check_config_uses_worker_package(tmp_path: Path) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    payload = edge_config_payload(
        camera_count=1,
        include_optional_fields=False,
        resident_ids=False,
    )
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    module = importlib.import_module("edge.runtime.edge_worker")

    assert module.main(["--config", str(config_path), "--check-config"]) == 0


def test_gpu_diagnostics_reports_nvml_success_and_binding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")

    class FakeNvml:
        @staticmethod
        def nvmlInit() -> None:
            pass

        @staticmethod
        def nvmlShutdown() -> None:
            pass

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> object:
            assert index == 0
            return object()

        @staticmethod
        def nvmlSystemGetDriverVersion() -> str:
            return "555.42"

        @staticmethod
        def nvmlDeviceGetName(handle: object) -> bytes:
            return b"NVIDIA Test"

    class FakeCuda:
        @staticmethod
        def init() -> None:
            pass

        @staticmethod
        def is_available() -> bool:
            return True

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml)
    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"cuda": FakeCuda})())
    success = module._gpu_diagnostics()
    assert success["nvml_available"] is True
    assert success["driver_version"] == "555.42"
    assert success["device_name"] == "NVIDIA Test"

    monkeypatch.setitem(sys.modules, "pynvml", None)
    failure = module._gpu_diagnostics()
    assert failure["nvml_available"] is False
    assert failure["nvml_error"] == "binding_unavailable"


def test_worker_entrypoint_fails_fast_when_profile_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0002: an unavailable requested profile exits non-zero without fallback."""
    config_path = tmp_path / "ml-worker.yaml"
    payload = edge_config_payload(camera_count=1, include_optional_fields=False, resident_ids=False)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    module = importlib.import_module("edge.runtime.edge_worker")

    def _fail(_stages: object) -> object:
        raise module.GlobalBootstrapError("profile_verify: no usable CUDA device")

    monkeypatch.setenv("ML_WORKER_PROFILE", "cuda")
    monkeypatch.setattr(module, "run_global_bootstrap", _fail)

    assert module.main(["--config", str(config_path)]) == 3


def test_profile_failure_publishes_gpu_diagnostics_before_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    published: list[dict[str, object]] = []

    class FakeSender:
        def __init__(
            self,
            relay_url: str,
            relay_token: str,
            diagnostics: object,
            facility_id: str,
            *,
            timeout_sec: float,
        ) -> None:
            assert relay_url == "http://relay.test"
            assert relay_token == "relay-token"
            assert facility_id == "facility-1"
            assert timeout_sec == 2.0
            self._diagnostics = diagnostics

        def publish_once(self) -> bool:
            published.append(self._diagnostics.to_payload("facility-1", None, 0))
            return True

    def _fail(_stages: object) -> object:
        raise module.GlobalBootstrapError("profile_verify: no usable CUDA device")

    monkeypatch.setenv("RELAY_URL", "http://relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setattr(module, "RuntimeStatusSender", FakeSender)
    monkeypatch.setattr(module, "run_global_bootstrap", _fail)
    monkeypatch.setattr(
        module,
        "_gpu_diagnostics",
        lambda: {
            "nvml_available": False,
            "cuda_context_ok": False,
            "driver_version": None,
            "device_name": None,
            "nvml_error": "query_failed:RuntimeError",
            "captured_at_sec": 1.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_load_startup_config",
        lambda _options: pytest.fail("profile failure must not load startup config"),
    )

    assert module.main([]) == 3
    assert len(published) == 1
    payload = published[0]
    assert payload["cameras"] == []
    assert payload["gpu"] == {
        "nvml_available": False,
        "cuda_context_ok": False,
        "driver_version": None,
        "device_name": None,
        "nvml_error": "query_failed:RuntimeError",
        "captured_at_sec": 1.0,
    }
    assert payload["worker"]["alive"] is False
    assert payload["worker"]["profile_boot_error"] == "profile_verify: no usable CUDA device"


def test_profile_failure_relay_unavailable_exits_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    attempts: list[float] = []

    def _fail(_stages: object) -> object:
        raise module.GlobalBootstrapError("profile_verify: no usable CUDA device")

    def _unavailable(_request: object, *, timeout: float) -> object:
        attempts.append(timeout)
        raise module.urllib.error.URLError("unavailable")

    monkeypatch.setenv("RELAY_URL", "http://relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setattr(module, "run_global_bootstrap", _fail)
    monkeypatch.setattr(module.urllib.request, "urlopen", _unavailable)
    monkeypatch.setattr(
        module,
        "_gpu_diagnostics",
        lambda: {
            "nvml_available": False,
            "cuda_context_ok": False,
            "driver_version": None,
            "device_name": None,
            "nvml_error": "binding_unavailable",
            "captured_at_sec": 1.0,
        },
    )

    started = time.monotonic()
    assert module.main([]) == 3
    assert time.monotonic() - started < 0.5
    assert attempts == [2.0]


def test_profile_failure_sender_exception_preserves_fail_fast_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")

    class FailingSender:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def publish_once(self) -> bool:
            raise RuntimeError("relay failed")

    def _fail(_stages: object) -> object:
        raise module.GlobalBootstrapError("profile_verify: no usable CUDA device")

    monkeypatch.setenv("RELAY_URL", "http://relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setattr(module, "RuntimeStatusSender", FailingSender)
    monkeypatch.setattr(module, "run_global_bootstrap", _fail)

    assert module.main([]) == 3
    assert "Profile boot diagnostics publish failed: RuntimeError" in capsys.readouterr().err


def test_profile_failure_malformed_relay_response_preserves_fail_fast_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")

    class MalformedResponse:
        def __enter__(self) -> MalformedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def _fail(_stages: object) -> object:
        raise module.GlobalBootstrapError("profile_verify: no usable CUDA device")

    def _malformed_response(_request: object, *, timeout: float) -> MalformedResponse:
        assert timeout == 2.0
        return MalformedResponse()

    monkeypatch.setenv("RELAY_URL", "http://relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-1")
    monkeypatch.setattr(module, "run_global_bootstrap", _fail)
    monkeypatch.setattr(module.urllib.request, "urlopen", _malformed_response)

    assert module.main([]) == 3
    assert "Profile boot diagnostics publish failed: AttributeError" in capsys.readouterr().err


def test_successful_profile_boot_keeps_existing_config_error_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")

    class SuccessfulBoot:
        outputs = {
            "profile_verify": module.BootContext(
                profile=PROFILE_REGISTRY["cpu"], device="cpu", decode="opencv"
            )
        }

    monkeypatch.setattr(module, "run_global_bootstrap", lambda _stages: SuccessfulBoot())
    monkeypatch.setattr(
        module,
        "_load_startup_config",
        lambda _options: (_ for _ in ()).throw(module.EdgeWorkerConfigError("config failed")),
    )

    def _unexpected_sender(*_args: object, **_kwargs: object) -> None:
        pytest.fail("normal profile boot must not publish failure status")

    monkeypatch.setattr(module, "RuntimeStatusSender", _unexpected_sender)

    assert module.main([]) == 2


def test_default_relay_promotes_valid_snapshot_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    sent: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def capture(request: object, *, timeout: float) -> Response:
        assert timeout == 0.5
        sent.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", capture)
    edge_event_id = "123e4567-e89b-42d3-a456-426614174000"
    snapshot = {
        "snapshot_id": edge_event_id,
        "path": module._snapshot_relative_path(
            "camera-1", "2026-07-28T12:00:00.000Z", edge_event_id
        ),
        "sha256": "a" * 64,
        "size_bytes": 42,
        "mime_type": "image/jpeg",
        "captured_at": "2026-07-28T12:00:00.000Z",
        "camera_id": "camera-1",
        "edge_event_id": edge_event_id,
    }

    module._RelayClient(
        alert_url="http://relay.test/alerts",
        heartbeat_url="http://relay.test/heartbeats",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="relay-token",
    ).emit(
        {
            "event_type": "fall",
            "probability": 0.9,
            "detected_at": "2026-07-28T12:00:00Z",
            "edge_event_id": edge_event_id,
            "snapshot": snapshot,
            "snapshot_jpeg": b"jpeg",
        }
    )

    assert sent[0]["snapshot"] == snapshot
    assert sent[0]["edge_event_id"] == edge_event_id
    assert sent[0]["snapshot_jpeg_base64"] == "anBlZw=="

    assert "snapshot" not in sent[0]["evidence"]
    assert "edge_event_id" not in sent[0]["evidence"]


def test_default_relay_allows_event_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    sent: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def capture(request: object, *, timeout: float) -> Response:
        del timeout
        sent.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", capture)
    edge_event_id = "123e4567-e89b-42d3-a456-426614174000"
    module._RelayClient(
        alert_url="http://relay.test/alerts",
        heartbeat_url="http://relay.test/heartbeats",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="relay-token",
    ).emit({"event_type": "fall", "probability": 0.9, "edge_event_id": edge_event_id})

    assert sent[0]["edge_event_id"] == edge_event_id
    assert "snapshot" not in sent[0]
    assert "snapshot" not in sent[0]["evidence"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "snapshots/../escape.jpg"),
        ("sha256", "A" * 64),
        ("size_bytes", 0),
        ("mime_type", "image/png"),
        ("captured_at", "2026-07-28T12:00:00Z"),
        ("camera_id", "camera-2"),
        ("edge_event_id", "not-a-uuid"),
        ("snapshot_id", "123e4567-e89b-42d3-a456-426614174001"),
    ],
)
def test_default_relay_rejects_malformed_present_snapshot(field: str, value: object) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    edge_event_id = "123e4567-e89b-42d3-a456-426614174000"
    snapshot: dict[str, object] = {
        "snapshot_id": edge_event_id,
        "path": module._snapshot_relative_path(
            "camera-1", "2026-07-28T12:00:00.000Z", edge_event_id
        ),
        "sha256": "a" * 64,
        "size_bytes": 42,
        "mime_type": "image/jpeg",
        "captured_at": "2026-07-28T12:00:00.000Z",
        "camera_id": "camera-1",
        "edge_event_id": edge_event_id,
    }
    snapshot[field] = value
    client = module._RelayClient(
        alert_url="http://relay.test/alerts",
        heartbeat_url="http://relay.test/heartbeats",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="relay-token",
    )

    with pytest.raises(ValueError):
        client.emit(
            {
                "event_type": "fall",
                "probability": 0.9,
                "edge_event_id": edge_event_id,
                "snapshot": snapshot,
            }
        )


@pytest.mark.parametrize("snapshot", [None, []])
def test_default_relay_rejects_non_object_snapshot(snapshot: object) -> None:

    module = importlib.import_module("edge.runtime.edge_worker")
    client = module._RelayClient(
        alert_url="http://relay.test/alerts",
        heartbeat_url="http://relay.test/heartbeats",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="relay-token",
    )

    with pytest.raises(TypeError, match="must be an object"):
        client.emit(
            {
                "event_type": "fall",
                "probability": 0.9,
                "edge_event_id": "123e4567-e89b-42d3-a456-426614174000",
                "snapshot": snapshot,
            }
        )


@pytest.mark.parametrize(
    ("snapshot_jpeg", "error"),
    [
        (None, TypeError),
        ("jpeg", TypeError),
        (b"", ValueError),
        (b"x" * (200 * 1024 + 1), ValueError),
    ],
)
def test_default_relay_rejects_invalid_inline_snapshot(
    snapshot_jpeg: object, error: type[Exception]
) -> None:
    module = importlib.import_module("edge.runtime.edge_worker")
    client = module._RelayClient(
        alert_url="http://relay.test/alerts",
        heartbeat_url="http://relay.test/heartbeats",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="relay-token",
    )

    with pytest.raises(error):
        client.emit(
            {
                "event_type": "fall",
                "probability": 0.9,
                "snapshot_jpeg": snapshot_jpeg,
            }
        )
