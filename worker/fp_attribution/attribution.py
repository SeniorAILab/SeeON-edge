"""Approved first-positive-match causal attribution over retained evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from worker.fp_attribution.evidence import AttributionEvidenceRecord
from worker.types.trace import DecisionTraceMissingReason, DecisionTraceReason, DecisionTraceState

_CORRELATION_SCHEMA = "fp-correlation-v1"
_FAULT_TOKEN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_STALE_REASONS = frozenset(
    {DecisionTraceReason.STALE_TRACK_EXIT.value, DecisionTraceReason.STALE_TRACK_CLEAR.value}
)
_POSE_REASONS = frozenset(
    {
        DecisionTraceReason.PERSON_OBSERVATION_MISSING.value,
        DecisionTraceReason.SCORE_MISSING.value,
    }
)
_POSE_MISSING = frozenset(
    {
        DecisionTraceMissingReason.NO_OBSERVED_PERSON.value,
        DecisionTraceMissingReason.NO_LIVE_CLASSIFIED_TRACK.value,
    }
)
_GEOMETRY_REASONS = frozenset(
    {
        DecisionTraceReason.BED_REGION_UNAVAILABLE.value,
        DecisionTraceReason.BED_OBSERVATION_MISSING.value,
        DecisionTraceReason.CONTAINED_IN_OTHER_BED.value,
        DecisionTraceReason.LIVE_GRACE_EXIT.value,
        DecisionTraceReason.ASSIGNED.value,
        DecisionTraceReason.ASSIGNMENT_HOLD.value,
        DecisionTraceReason.BELOW_CONTAINMENT.value,
    }
)
_DELIVERY_KINDS = frozenset({"BACKEND_OR_UI_DUPLICATE", "DELIVERY_RETRY"})
_KIND_KEYS = {
    "BACKEND_OR_UI_DUPLICATE": frozenset(
        {"schema", "edge_event_id", "kind", "user_visible_delivery_count"}
    ),
    "DELIVERY_RETRY": frozenset(
        {"schema", "edge_event_id", "kind", "user_visible_delivery_count"}
    ),
    # Transport remains an annotation/export kind, never a causal category.
    "TRANSPORT_ONLY": frozenset(
        {"schema", "edge_event_id", "kind", "user_visible_delivery_count"}
    ),
    "CAMERA_LIGHTING_OR_DECODE": frozenset(
        {"schema", "edge_event_id", "kind", "typed_fault_code"}
    ),
}

AttributionCategory = Literal[
    "BACKEND_OR_UI_DUPLICATE",
    "DELIVERY_RETRY",
    "BED_STALE_TRACK",
    "FALL_LATCH_REARM",
    "EPISODE_FRAGMENTATION",
    "TRACKER_OR_IDENTITY",
    "ZERO_OR_MISSING_POSE",
    "BED_GEOMETRY_OR_ASSIGNMENT",
    "CAMERA_LIGHTING_OR_DECODE",
    "MODEL_OR_THRESHOLD",
    "ANNOTATION_ERROR",
]
CorrelationStatus = Literal["absent", "accepted", "rejected"]
RESERVED_MECHANISMS: tuple[str, ...] = ("IDLE_STATIC",)


class PredicateVerdict(StrEnum):
    MATCH = "match"
    SKIP = "skip"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class AttributionAnnotations:
    matched_predicate: str | None
    coverage_status: str
    coverage_reason: str | None
    neighborhood_pruned: bool
    attempt_count: int
    backend_event_ids: tuple[str, ...]
    correlation_status: CorrelationStatus
    correlation_kind: str | None
    correlation_rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class AttributionDecision:
    category: AttributionCategory | None
    evidence_status: str
    annotations: AttributionAnnotations


@dataclass(frozen=True, slots=True)
class _Correlation:
    status: CorrelationStatus
    kind: str | None
    rejection_reason: str | None
    delivery_count: int | None
    typed_fault_code: str | None


@dataclass(frozen=True, slots=True)
class _Context:
    record: AttributionEvidenceRecord
    correlation: _Correlation


PredicateFn = Callable[[_Context], PredicateVerdict]


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    category: AttributionCategory
    evaluate: PredicateFn


def classify_record(
    record: AttributionEvidenceRecord,
    *,
    correlation_export: object | None = None,
) -> AttributionDecision:
    correlation = _parse_correlation(record.edge_event_id, correlation_export)
    if (
        record.evidence_status != "COMPLETE"
        or record.neighborhood_pruned
        or not record.prevented_eligible
    ):
        return _decision(record, None, None, correlation, record.evidence_status)

    # Supplied but rejected correlation is an applicable, unevaluable claim;
    # accepted single-delivery transport is annotation only. Neither may fall
    # through to a weaker positive category.
    if correlation.status == "rejected" or (
        correlation.status == "accepted" and correlation.kind == "TRANSPORT_ONLY"
    ):
        return _decision(record, None, None, correlation, "UNKNOWN")

    context = _Context(record=record, correlation=correlation)
    for spec in PREDICATE_REGISTRY:
        verdict = spec.evaluate(context)
        if verdict is PredicateVerdict.MATCH:
            return _decision(record, spec.category, spec.category, correlation, "COMPLETE")
        if verdict is PredicateVerdict.INSUFFICIENT:
            return _decision(record, None, None, correlation, "UNKNOWN")
    return _decision(record, None, None, correlation, "UNKNOWN")


def machine_bytes(decision: AttributionDecision) -> bytes:
    annotations = decision.annotations
    payload = {
        "annotations": {
            "attempt_count": annotations.attempt_count,
            "backend_event_ids": list(annotations.backend_event_ids),
            "correlation_kind": annotations.correlation_kind,
            "correlation_rejection_reason": annotations.correlation_rejection_reason,
            "correlation_status": annotations.correlation_status,
            "coverage_reason": annotations.coverage_reason,
            "coverage_status": annotations.coverage_status,
            "matched_predicate": annotations.matched_predicate,
            "neighborhood_pruned": annotations.neighborhood_pruned,
        },
        "category": decision.category,
        "evidence_status": decision.evidence_status,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decision(
    record: AttributionEvidenceRecord,
    category: AttributionCategory | None,
    matched: str | None,
    correlation: _Correlation,
    evidence_status: str,
) -> AttributionDecision:
    return AttributionDecision(
        category=category,
        evidence_status=evidence_status,
        annotations=AttributionAnnotations(
            matched_predicate=matched,
            coverage_status=record.coverage_status,
            coverage_reason=record.coverage_reason,
            neighborhood_pruned=record.neighborhood_pruned,
            attempt_count=record.attempt_count,
            backend_event_ids=record.backend_event_ids,
            correlation_status=correlation.status,
            correlation_kind=correlation.kind,
            correlation_rejection_reason=correlation.rejection_reason,
        ),
    )


def _parse_correlation(edge_event_id: str, export: object | None) -> _Correlation:
    if export is None:
        return _Correlation("absent", None, None, None, None)
    if not isinstance(export, dict):
        return _rejected("malformed_export")
    schema = export.get("schema")
    kind = export.get("kind")
    if schema != _CORRELATION_SCHEMA:
        return _rejected("schema_invalid")
    if not isinstance(kind, str) or kind not in _KIND_KEYS:
        return _rejected("kind_not_allowlisted")
    if set(export) != _KIND_KEYS[kind]:
        return _rejected("extra_or_missing_fields")
    if export.get("edge_event_id") != edge_event_id:
        return _rejected("event_id_mismatch")
    if kind == "CAMERA_LIGHTING_OR_DECODE":
        fault = export.get("typed_fault_code")
        if not isinstance(fault, str) or _FAULT_TOKEN.fullmatch(fault) is None:
            return _rejected("typed_fault_missing")
        return _Correlation("accepted", kind, None, None, fault)
    count = export.get("user_visible_delivery_count")
    if type(count) is not int:
        return _rejected("count_not_integer")
    if kind in _DELIVERY_KINDS and count < 2:
        return _rejected("count_not_repeated")
    if kind == "TRANSPORT_ONLY" and count != 1:
        return _rejected("count_not_transport_only")
    return _Correlation("accepted", kind, None, count, None)


def _rejected(reason: str) -> _Correlation:
    return _Correlation("rejected", None, reason, None, None)


def _aligned(record: AttributionEvidenceRecord) -> bool:
    alignment = record.domain_alignment
    return (
        alignment.status == "ALIGNED"
        and alignment.same_track is True
        and alignment.same_domain is True
        and alignment.same_camera_boot_epoch is True
        and record.boot_changed is False
        and record.epoch_changed is False
    )


def _backend_or_ui_duplicate(context: _Context) -> PredicateVerdict:
    correlation = context.correlation
    if correlation.status != "accepted" or correlation.kind != "BACKEND_OR_UI_DUPLICATE":
        return PredicateVerdict.SKIP
    if (
        correlation.delivery_count is not None
        and correlation.delivery_count >= 2
        and len(set(context.record.backend_event_ids)) >= 2
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _delivery_retry(context: _Context) -> PredicateVerdict:
    correlation = context.correlation
    if correlation.status != "accepted" or correlation.kind != "DELIVERY_RETRY":
        return PredicateVerdict.SKIP
    if (
        correlation.delivery_count is not None
        and correlation.delivery_count >= 2
        and context.record.attempt_count >= 2
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _bed_stale_track(context: _Context) -> PredicateVerdict:
    record = context.record
    if record.decision_reason not in _STALE_REASONS:
        return PredicateVerdict.SKIP
    stale = record.track_staleness
    if (
        _aligned(record)
        and stale.same_track is True
        and stale.last_seen_offset_frames is not None
        and stale.last_seen_offset_frames > 0
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _fall_latch_rearm(context: _Context) -> PredicateVerdict:
    record = context.record
    if record.decision_reason != DecisionTraceReason.FALL_ONSET.value:
        return PredicateVerdict.SKIP
    latch = record.fall_latch
    if latch.status != "AVAILABLE" and not record.associated_sibling_event_ids:
        return PredicateVerdict.SKIP
    if (
        record.previous_state == DecisionTraceState.CLEAR.value
        and record.current_state == DecisionTraceState.FALL.value
        and _aligned(record)
        and latch.status == "AVAILABLE"
        and latch.same_track is True
        and latch.same_domain is True
        and latch.rise_before_rearm is True
        and latch.rearm_frames is not None
        and latch.rearm_frames >= 0
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _episode_fragmentation(context: _Context) -> PredicateVerdict:
    record = context.record
    if not record.associated_sibling_event_ids:
        return PredicateVerdict.SKIP
    distinct_ids = {record.edge_event_id, *record.associated_sibling_event_ids}
    latch = record.fall_latch
    latch_explains = (
        latch.status == "AVAILABLE"
        and latch.same_track is True
        and latch.same_domain is True
        and latch.rise_before_rearm is True
    )
    if len(distinct_ids) >= 2 and _aligned(record) and not latch_explains:
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _tracker_or_identity(context: _Context) -> PredicateVerdict:
    record = context.record
    if not record.track_changed:
        return PredicateVerdict.SKIP
    # The immediate prior-frame same-track fact plus aligned neighborhood churn
    # is the available positive consecutive-eligible-frame proof. The summary
    # boolean alone never matches.
    stale = record.track_staleness
    if _aligned(record) and stale.same_track is True and stale.last_seen_offset_frames == 0:
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _zero_or_missing_pose(context: _Context) -> PredicateVerdict:
    record = context.record
    applicable = (
        record.person_presence.status == "PERSON_GAP"
        or record.decision_reason in _POSE_REASONS
        or record.track_missing_reason in _POSE_MISSING
        or record.score_missing_reason in _POSE_MISSING
    )
    if not applicable:
        return PredicateVerdict.SKIP
    presence = record.person_presence
    if (
        _aligned(record)
        and presence.status == "PERSON_GAP"
        and presence.duration_frames is not None
        and presence.duration_frames >= 30
        and record.due_signal.status is not None
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _bed_geometry_or_assignment(context: _Context) -> PredicateVerdict:
    record = context.record
    if not (record.bed_changed or record.decision_reason in _GEOMETRY_REASONS):
        return PredicateVerdict.SKIP
    bed = record.bed_state
    if (
        _aligned(record)
        and bed.status == "AVAILABLE"
        and bed.sequence is not None
        and len(bed.sequence) >= 2
        and bed.durations_frames is not None
        and len(bed.sequence) == len(bed.durations_frames)
        and bed.same_track is True
        and bed.same_domain is True
    ):
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _camera_lighting_or_decode(context: _Context) -> PredicateVerdict:
    correlation = context.correlation
    if correlation.status == "accepted" and correlation.kind == "CAMERA_LIGHTING_OR_DECODE":
        return PredicateVerdict.MATCH
    return PredicateVerdict.SKIP


def _model_or_threshold(context: _Context) -> PredicateVerdict:
    record = context.record
    if record.score is not None and record.threshold is not None:
        return PredicateVerdict.MATCH
    return PredicateVerdict.INSUFFICIENT


def _annotation_error(_context: _Context) -> PredicateVerdict:
    # No independent adjudication export is available in the current seam.
    return PredicateVerdict.SKIP


PREDICATE_REGISTRY: tuple[PredicateSpec, ...] = (
    PredicateSpec("BACKEND_OR_UI_DUPLICATE", _backend_or_ui_duplicate),
    PredicateSpec("DELIVERY_RETRY", _delivery_retry),
    PredicateSpec("BED_STALE_TRACK", _bed_stale_track),
    PredicateSpec("FALL_LATCH_REARM", _fall_latch_rearm),
    PredicateSpec("EPISODE_FRAGMENTATION", _episode_fragmentation),
    PredicateSpec("TRACKER_OR_IDENTITY", _tracker_or_identity),
    PredicateSpec("ZERO_OR_MISSING_POSE", _zero_or_missing_pose),
    PredicateSpec("BED_GEOMETRY_OR_ASSIGNMENT", _bed_geometry_or_assignment),
    PredicateSpec("CAMERA_LIGHTING_OR_DECODE", _camera_lighting_or_decode),
    PredicateSpec("MODEL_OR_THRESHOLD", _model_or_threshold),
    PredicateSpec("ANNOTATION_ERROR", _annotation_error),
)

__all__ = [
    "AttributionAnnotations",
    "AttributionCategory",
    "AttributionDecision",
    "PREDICATE_REGISTRY",
    "PredicateSpec",
    "RESERVED_MECHANISMS",
    "classify_record",
    "machine_bytes",
]
