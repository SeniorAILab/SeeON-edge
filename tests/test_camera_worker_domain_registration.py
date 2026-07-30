from __future__ import annotations

import ast
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from edge_worker_fixtures import edge_config_payload
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contracts.frame import Frame
from edge.domains import DOMAIN_REGISTRY, DomainRegistration
from edge.domains.base import DomainDetector
from edge.domains.bed_exit.schema import DomainDebugSnapshot
from edge.perception.domain_input import DomainInput
from edge.runtime import edge_worker
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker import _RelayClient
from edge.runtime.edge_worker_config import EdgeWorkerConfig
from edge.runtime.scheduler import Scheduler


@dataclass
class _Sink:
    events: list[dict[str, object]]

    def emit(self, event: dict[str, object]) -> None:
        self.events.append(event)
class _DummyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sensitivity: int = Field(gt=0)


class _FakeHTTPResponse:
    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return b"{}"


def _dummy_registration(factory: object) -> DomainRegistration:
    return DomainRegistration(
        name="dummy",
        factory=factory,
        enabled=True,
        input_view="dummy-view",
        event_types=frozenset({"dummy-alert"}),
        debug_snapshot_adapter=_dummy_debug,
        input_preparer=_dummy_prepare,
        config_schema=_DummyConfig,
    )




class _DummyDetector(DomainDetector):
    enabled = True

    def __init__(self) -> None:
        self.seen: DomainInput | None = None

    def update(self, domain_input: DomainInput) -> tuple[dict[str, object], ...]:
        self.seen = domain_input
        return ({"domain": "dummy", "event_type": "dummy-alert", "identity": 1},)


def _dummy_debug(detector: DomainDetector, _frame_index: int) -> DomainDebugSnapshot | None:
    assert isinstance(detector, _DummyDetector)
    return None
def _dummy_prepare(
    detector: DomainDetector, input_view: str, domain_input: DomainInput
) -> DomainInput:
    assert isinstance(detector, _DummyDetector)
    detector.prepared_view = input_view
    return domain_input



def test_registry_only_domain_can_be_enabled_in_yaml_and_composed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _dummy_registration(lambda _config, _model: _DummyDetector())
    monkeypatch.setitem(DOMAIN_REGISTRY, "dummy", registration)
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {"enabled": ["dummy"]}
    config = EdgeWorkerConfig.model_validate(payload)
    detectors = edge_worker._domain_detectors(config)
    detector = next(detector for detector in detectors if isinstance(detector, _DummyDetector))
    sink = _Sink([])
    worker = CameraWorker(
        camera_id="camera",
        facility_id="facility",
        frame_source=(),
        runners={},
        scheduler=Scheduler({}),
        domain_detectors=detectors,
        event_sink=sink,
    )

    worker.process_frame(Frame(0, 1.5, np.zeros((4, 6, 3), dtype=np.uint8)))

    assert len(detectors) == 1
    assert detector.seen is not None
    assert (detector.seen.frame_width, detector.seen.frame_height) == (6, 4)
    assert detector.prepared_view == "dummy-view"
    assert sink.events[0]["event_type"] == "dummy-alert"


def test_registry_domain_event_reaches_relay_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        del timeout
        captured.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse()

    monkeypatch.setitem(
        DOMAIN_REGISTRY,
        "dummy",
        _dummy_registration(lambda _config, _model: _DummyDetector()),
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {"enabled": ["dummy"]}
    config = EdgeWorkerConfig.model_validate(payload)
    client = _RelayClient(
        alert_url="http://127.0.0.1:8000/api/v1/relay/alerts",
        heartbeat_url="http://127.0.0.1:8000/api/v1/relay/heartbeat",
        camera_id="camera",
        facility_id="facility",
        resident_id=None,
        relay_token="token",
    )
    worker = CameraWorker(
        camera_id="camera",
        facility_id="facility",
        frame_source=(),
        runners={},
        scheduler=Scheduler({}),
        domain_detectors=edge_worker._domain_detectors(config),
        event_sink=client,
    )

    worker.process_frame(Frame(0, 1.5, np.zeros((4, 6, 3), dtype=np.uint8)))

    assert captured[0]["event_type"] == "dummy-alert"
    evidence = captured[0]["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["domain"] == "dummy"
    assert evidence["event_type"] == "dummy-alert"
    assert evidence["identity"] == 1


def test_relay_client_rejects_unregistered_event_type() -> None:
    client = _RelayClient(
        alert_url="http://127.0.0.1:8000/api/v1/relay/alerts",
        heartbeat_url="http://127.0.0.1:8000/api/v1/relay/heartbeat",
        camera_id="camera",
        facility_id="facility",
        resident_id=None,
        relay_token="token",
        allowed_event_types=frozenset({"dummy-alert"}),
    )

    with pytest.raises(ValueError, match="unregistered domain event type: 'unregistered'"):
        client.emit({"event_type": "unregistered"})


def test_enabled_domains_rejects_unregistered_name() -> None:
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {"enabled": ["typo"]}

    with pytest.raises(ValidationError, match="domains.enabled contains unknown domain: typo"):
        EdgeWorkerConfig.model_validate(payload)

@pytest.mark.parametrize(
    "enabled, message",
    [
        (["fall", "fall"], "domains.enabled contains duplicate domain: fall"),
        (["FALL"], "domains.enabled contains unknown domain: FALL"),
        ([" fall "], "domains.enabled contains unknown domain:  fall "),
        ([""], "domains.enabled contains unknown domain: "),
    ],
)
def test_enabled_domains_rejects_duplicate_and_variant_names(
    enabled: list[str], message: str
) -> None:
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {"enabled": enabled}

    with pytest.raises(ValidationError, match=message):
        EdgeWorkerConfig.model_validate(payload)


def test_registry_domain_config_is_passed_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object] | None] = []

    def factory(config: object | None, _model: object | None) -> DomainDetector:
        assert config is None or isinstance(config, dict)
        received.append(config)
        return _DummyDetector()

    monkeypatch.setitem(DOMAIN_REGISTRY, "dummy", _dummy_registration(factory))
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {"dummy": {"enabled": True, "sensitivity": 3}}

    detectors = edge_worker._domain_detectors(EdgeWorkerConfig.model_validate(payload))

    assert len(detectors) == 1
    assert received == [{"sensitivity": 3}]


def test_camera_worker_has_no_domain_specific_bypasses() -> None:
    source = Path("edge/runtime/camera_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_attributes = {"fall_classifier", "observation_enricher"}
    forbidden_calls = {"resolve_bed_regions"}
    forbidden_names = {"BedExitDebugSnapshot"}
    forbidden_literals = {"fall", "bed-exit"}

    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    ]
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
    ]
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in forbidden_names
    ]
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in forbidden_literals
    ]
