from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

import worker.domains.registry as registry_module
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.domains.registry import DomainRegistration
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.runtime.config import WorkerConfig
from worker.types import BusinessEvent, DecisionInput


def _worker_config_payload(**domains: object) -> dict[str, object]:
    return {
        "relay": {"url": "http://relay.test", "token": "relay-token"},
        "domains": domains,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://example.test/camera-1",
            }
        ],
    }


def test_worker_config_rejects_duplicate_enabled_domain_names() -> None:
    with pytest.raises(ValidationError, match="domains.enabled contains duplicate domain: fall"):
        WorkerConfig.model_validate(_worker_config_payload(enabled=["fall", "fall"]))


def test_domain_registry_extension_composes_through_event_aggregator_generically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[DecisionInput] = []

    @dataclass(slots=True)
    class _DummyDetector:
        def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
            seen.append(input_value)
            return (
                BusinessEvent(
                    domain="dummy",
                    event_type="dummy-alert",
                    identity=1,
                    camera_id="camera-1",
                    facility_id="facility-1",
                    time_sec=input_value.time_sec,
                    probability=1.0,
                ),
            )

    dummy_registration = DomainRegistration(
        domain="dummy",
        input_view="dummy-view",
        event_types=frozenset({"dummy-alert"}),
        factory=lambda _dependencies: _DummyDetector(),
        requires=frozenset(),
    )
    extended = dict(registry_module.DOMAIN_REGISTRY)
    extended["dummy"] = dummy_registration
    monkeypatch.setattr(registry_module, "DOMAIN_REGISTRY", MappingProxyType(extended))

    detector = registry_module.DOMAIN_REGISTRY["dummy"].factory(None)
    aggregator = EventAggregator(
        deciders=(detector,),
        incidents=IncidentManager(identity_path=tmp_path / "identities.jsonl"),
    )
    frame = DecisionInput(
        observation=FrameObservation(),
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=1.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )

    admitted = aggregator.update(frame)

    assert len(seen) == 1
    assert seen[0] is frame
    assert len(admitted) == 1
    assert admitted[0].domain == "dummy"
    assert admitted[0].event_type == "dummy-alert"
    assert "dummy" in registry_module.list_domains()
    assert "dummy" in registry_module.enabled_domains()


def test_camera_worker_has_no_domain_specific_bypasses() -> None:
    source = Path("worker/runtime/worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {"FallEventLatch", "BedExitMonitor"}
    forbidden_attributes = {
        "classifier",
        "_previous_fall",
        "last_debug_snapshot",
        "_assignments",
        "_latch",
    }

    name_hits = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in forbidden_names
    ]
    attr_hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    ]

    assert not name_hits, [ast.dump(node) for node in name_hits]
    assert not attr_hits, [ast.dump(node) for node in attr_hits]
