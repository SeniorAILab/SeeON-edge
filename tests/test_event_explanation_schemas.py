"""Characterization and contract tests for Event Explanation schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.features.clips.schemas import ClipAnalysisResponse
from backend.app.features.evidence.explanation_schemas import EventExplanationResponse
from backend.app.features.evidence.operator_router import IncidentReviewRequest

_TRACE_ID = "a" * 64
_POLICY_ID = "b" * 64
_MANIFEST_SHA = "c" * 64

_CLIP_ANALYSIS_PAYLOAD = {
    "clip_id": "clip-a",
    "decision_trace_id": _TRACE_ID,
    "module_qualified_id": "fall.v1",
    "policy_qualified_id": "fall.policy.v1",
    "effective_policy_id": _POLICY_ID,
    "runtime_manifest_sha256": _MANIFEST_SHA,
    "reason": "fall-onset",
    "previous_state": "clear",
    "current_state": "fall",
    "triggered": True,
    "track_id": 1,
    "bed_id": None,
    "values": [{"name": "fall_probability", "value": 0.9, "missing_reason": None}],
}

_INCIDENT_REVIEW_PAYLOAD = {
    "expected_version": 1,
    "disposition": "FALSE_POSITIVE",
}

_FORBIDDEN_RAW_FIELDS = (
    "payload_json",
    "path",
    "polygon",
    "coordinates",
)


def test_existing_clip_and_evidence_schemas_reject_extra_fields() -> None:
    # Given: the existing clip analysis and evidence review Pydantic contracts.
    clip = ClipAnalysisResponse.model_validate(_CLIP_ANALYSIS_PAYLOAD)
    review = IncidentReviewRequest.model_validate(_INCIDENT_REVIEW_PAYLOAD)

    # When: those models are constructed from allowlisted fields only.
    clip_dump = clip.model_dump(mode="json")
    review_dump = review.model_dump(mode="json")

    # Then: extra="forbid" already rejects raw/extra keys on both contracts.
    assert clip_dump["decision_trace_id"] == _TRACE_ID
    assert review_dump["disposition"] == "FALSE_POSITIVE"
    for field_name in _FORBIDDEN_RAW_FIELDS:
        with pytest.raises(ValidationError) as clip_error:
            ClipAnalysisResponse.model_validate({**_CLIP_ANALYSIS_PAYLOAD, field_name: "sentinel"})
        assert clip_error.value.errors()[0]["type"] == "extra_forbidden"
        with pytest.raises(ValidationError) as review_error:
            IncidentReviewRequest.model_validate(
                {**_INCIDENT_REVIEW_PAYLOAD, field_name: "sentinel"}
            )
        assert review_error.value.errors()[0]["type"] == "extra_forbidden"


_SHA256 = "c" * 64
_TRACE = "a" * 64
_BOOT = "boot-1"
_EDGE_EVENT_ID = "11111111-1111-4111-8111-111111111111"

_REQUESTED_FIELD_NAMES = frozenset(
    {
        "edge_event_id",
        "facility_id",
        "camera_id",
        "domain",
        "event_type",
        "detected_at",
        "worker_boot_id",
        "stream_epoch",
        "frame_seq",
        "decision_trace_id",
        "reason",
        "previous_state",
        "current_state",
        "probability",
        "threshold",
        "decision_values",
        "missing_values",
        "track_id",
        "bed_id",
        "config_version",
        "policy_qualified_id",
        "model",
        "detector_version",
        "runtime_manifest_sha256",
        "worker_build_revision",
        "image_revision",
        "outbox_state",
        "attempt_count",
        "last_delivery_disposition",
        "last_http_status",
        "backend_event_id",
        "snapshot",
        "clip",
        "decision_provenance",
        "delivery",
        "media",
        "review",
        "neighborhood",
        "correlation",
    }
)


def _present(value: object) -> dict[str, object]:
    return {"value": value, "missing_reason": None}


def _missing(reason: str) -> dict[str, object]:
    return {"value": None, "missing_reason": reason}


def _complete_payload() -> dict[str, object]:
    return {
        "decision_provenance": "COMPLETE",
        "decision_provenance_reasons": [],
        "edge_event_id": _EDGE_EVENT_ID,
        "facility_id": _missing("facility_id_not_a_first_class_column"),
        "camera_id": "camera-1",
        "domain": "fall",
        "event_type": "fall",
        "detected_at": "2026-08-15T00:00:00Z",
        "worker_boot_id": _present(_BOOT),
        "stream_epoch": _present(0),
        "frame_seq": _present(100),
        "decision_trace_id": _present(_TRACE),
        "reason": _present("fall-onset"),
        "previous_state": _present("clear"),
        "current_state": _present("fall"),
        "triggered": _present(True),
        "probability": _present(0.91),
        "threshold": _present(0.5),
        "decision_values": [
            {"name": "fall_probability", "value": 0.91},
            {"name": "operating_threshold", "value": 0.5},
        ],
        "missing_values": [
            {"name": "containment_ratio", "missing_reason": "domain_inapplicable"},
        ],
        "track_id": _present(7),
        "bed_id": _missing("domain_inapplicable"),
        "config_version": _present(3),
        "policy_qualified_id": _present("fall.policy.v1"),
        "model": _present("onnxruntime"),
        "detector_version": _present("detector-v1"),
        "runtime_manifest_sha256": _present(_SHA256),
        "worker_build_revision": _present("d" * 40),
        "image_revision": _present("e" * 40),
        "delivery": {
            "status": "COMPLETE",
            "reasons": [],
            "outbox_state": _present("ACKED"),
            "attempt_count": _present(1),
            "last_delivery_disposition": _missing("disposition_not_persisted"),
            "last_http_status": _missing("last_http_status_not_persisted"),
            "backend_event_id": _present("backend-event-1"),
        },
        "media": {
            "status": "COMPLETE",
            "reasons": [],
            "snapshot": {"state": "AVAILABLE", "missing_reason": None},
            "clip": {"state": "AVAILABLE", "missing_reason": None},
        },
        "review": {
            "status": "UNAVAILABLE",
            "reasons": ["review_not_recorded"],
            "disposition": _missing("review_not_recorded"),
        },
        "neighborhood": {
            "status": "COMPLETE",
            "reasons": [],
            "neighborhood_pruned": False,
            "retained_frame_count": 30,
        },
        "correlation": {
            "status": "UNAVAILABLE",
            "reasons": ["alert_correlation_export_not_supplied"],
            "alert_id": _missing("alert_correlation_export_not_supplied"),
        },
    }


def _json_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(_json_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_json_keys(child))
    return keys


def test_event_explanation_contract_rejects_invalid_and_preserves_fields() -> None:
    # Given: a COMPLETE decision explanation with typed domain-inapplicable nulls.
    payload = _complete_payload()

    # When: the response contract validates and serializes that payload.
    parsed = EventExplanationResponse.model_validate(payload)
    dumped = parsed.model_dump(mode="json")

    # Then: requested machine-consumed names serialize, extra/raw/unknown/silent-null
    # inputs reject, and domain-inapplicable nulls do not downgrade COMPLETE.
    assert dumped["decision_provenance"] == "COMPLETE"
    assert dumped["facility_id"]["missing_reason"] == "facility_id_not_a_first_class_column"
    assert dumped["delivery"]["last_http_status"]["missing_reason"] == (
        "last_http_status_not_persisted"
    )
    assert dumped["missing_values"][0]["name"] == "containment_ratio"
    assert dumped["missing_values"][0]["missing_reason"] == "domain_inapplicable"
    assert dumped["bed_id"]["missing_reason"] == "domain_inapplicable"
    assert dumped["review"]["status"] == "UNAVAILABLE"
    assert dumped["correlation"]["status"] == "UNAVAILABLE"
    assert _REQUESTED_FIELD_NAMES <= _json_keys(dumped)
    for field_name in (*_FORBIDDEN_RAW_FIELDS, "notes", "actor_id"):
        with pytest.raises(ValidationError) as extra_error:
            EventExplanationResponse.model_validate({**payload, field_name: "sentinel"})
        assert extra_error.value.errors()[0]["type"] == "extra_forbidden"
    with pytest.raises(ValidationError):
        EventExplanationResponse.model_validate({**payload, "decision_provenance": "UNKNOWN"})
    with pytest.raises(ValidationError):
        EventExplanationResponse.model_validate(
            {
                **payload,
                "facility_id": _missing("because the camera was dark"),
            }
        )
    silent_null = _complete_payload()
    silent_null["probability"] = {"value": None, "missing_reason": None}
    with pytest.raises(ValidationError):
        EventExplanationResponse.model_validate(silent_null)
    both_set = _complete_payload()
    both_set["bed_id"] = {"value": 1, "missing_reason": "domain_inapplicable"}
    with pytest.raises(ValidationError):
        EventExplanationResponse.model_validate(both_set)
