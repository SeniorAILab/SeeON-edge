from __future__ import annotations

from contracts import EventApiPayload
from contracts.event import (
    DetectionEventType,
    Level,
    Severity,
    front_event_type,
    register_event_type,
)
from shared.events.schemas import build_emitted_event


def test_emitted_event_schema_has_required_fields_and_contract_enums() -> None:
    event = build_emitted_event(
        facility="facility-001",
        camera="cam-001",
        domain="fall_detection",
        event_type="fall",
        severity=Level.HIGH,
        evidence={"confidence": 0.91},
    )

    assert event.as_dict() == {
        "facility": "facility-001",
        "camera": "cam-001",
        "domain": "fall_detection",
        "event_type": "fall",
        "lifecycle": "detected",
        "severity": "HIGH",
        "front_event_type": "FALL_RISK",
        "evidence": {"confidence": 0.91},
    }
    assert Severity is Level


def test_event_type_registry_is_open_and_defaults_unknown_to_other() -> None:
    assert front_event_type("new-detection") is DetectionEventType.OTHER

    register_event_type("new-detection", DetectionEventType.WANDERING)

    assert front_event_type("new-detection") is DetectionEventType.WANDERING
    event = build_emitted_event(
        facility="facility-001",
        camera="cam-001",
        domain="behavior",
        event_type="new-detection",
        severity="MEDIUM",
    )
    assert event.front_event_type is DetectionEventType.WANDERING


def test_front_event_type_mapping_matches_front_detection_event_type_values() -> None:
    assert [item.value for item in Level] == ["LOW", "MEDIUM", "HIGH"]
    assert [item.value for item in DetectionEventType] == [
        "STABLE",
        "MOVEMENT_INCREASE",
        "REPEATED_STANDING_ATTEMPT",
        "FALL_RISK",
        "SOLO_MOVEMENT",
        "PROLONGED_INACTIVITY",
        "WANDERING",
        "BED_EXIT",
        "OTHER",
    ]
    assert front_event_type("fall") is DetectionEventType.FALL_RISK
    assert front_event_type("bed-exit") is DetectionEventType.BED_EXIT
    assert front_event_type("detection-lost") is DetectionEventType.OTHER


def test_event_api_payload_matches_backend_contract() -> None:
    payload = EventApiPayload(
        camera_id="camera-1",
        type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        confidence=0.91,
    )

    assert payload.as_dict() == {
        "camera_id": "camera-1",
        "type": "fall",
        "detected_at": "2026-06-23T12:00:00.000Z",
        "confidence": 0.91,
    }
