"""Compose one Event Explanation response from committed Foundation seams."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, TypeGuard, overload

from pydantic import ValidationError

from backend.app.features.evidence.explanation_manifest import (
    RuntimeManifestMissingReason,
    RuntimeManifestProjection,
    project_runtime_manifest,
)
from backend.app.features.evidence.explanation_neighborhood import (
    CoverageReason,
    EventNeighborhoodQuery,
    NeighborhoodCoverage,
)
from backend.app.features.evidence.explanation_schemas import (
    AnalysisMissingReason,
    BedIdFact,
    ConfigVersionFact,
    DecisionProvenance,
    DecisionProvenanceReason,
    DecisionReason,
    DecisionReasonFact,
    DecisionState,
    DecisionStateFact,
    DecisionTraceIdFact,
    DecisionTraceMissingReason,
    DecisionValueMissingReason,
    DecisionValueName,
    DetectorVersionFact,
    EventDomain,
    EventExplanationDecisionValue,
    EventExplanationMissingValue,
    EventExplanationNeighborhood,
    EventExplanationResponse,
    EventExplanationReview,
    EventType,
    FacilityIdFact,
    FrameSeqFact,
    ModelFact,
    NeighborhoodMissingReason,
    PolicyQualifiedIdFact,
    ProbabilityFact,
    ReviewDispositionFact,
    ReviewDispositionToken,
    RuntimeManifestSha256Fact,
    RuntimeMissingReason,
    SourceRevisionFact,
    StreamEpochFact,
    ThresholdFact,
    TrackBedMissingReason,
    TrackIdFact,
    TriggeredFact,
    WorkerBootIdFact,
)
from backend.app.features.evidence.explanation_sections import project_explanation_sections
from backend.app.features.evidence.explanation_store import (
    EventExplanationCurrentReview,
    EventExplanationDecision,
    EventExplanationIdentity,
    EventExplanationQuery,
    EventExplanationRuntime,
    TraceRefConflict,
)
from shared.edge_db.connection import RuntimeActor, open_runtime_database

_REVIEW_DISPOSITIONS: frozenset[ReviewDispositionToken] = frozenset(
    {"TRUE_POSITIVE", "FALSE_POSITIVE"}
)

_DECISION_REASONS: frozenset[DecisionReason] = frozenset(
    {
        "trace-unavailable",
        "outside-detection-window",
        "score-missing",
        "fall-onset",
        "fall-active",
        "below-threshold",
        "bed-region-unavailable",
        "bed-observation-missing",
        "stale-track-exit",
        "stale-track-clear",
        "assigned",
        "assignment-hold",
        "below-containment",
        "contained",
        "contained-in-other-bed",
        "live-grace-exit",
        "live-grace",
        "person-observation-missing",
    }
)
_DECISION_STATES: frozenset[DecisionState] = frozenset(
    {
        "unknown",
        "not-evaluated",
        "no-decision",
        "clear",
        "fall",
        "live-grace",
        "contained",
        "triggered",
        "retired",
        "unassigned",
        "other-bed",
    }
)
_VALUE_NAMES: frozenset[DecisionValueName] = frozenset(
    {
        "operating_threshold",
        "window_frames",
        "fall_probability",
        "containment_ratio",
        "max_other_containment_ratio",
        "min_containment",
        "candidate_frames",
        "hold_frames_threshold",
        "grace_frames_before",
        "grace_frames_after",
        "grace_threshold",
        "bed_id",
        "decision_state",
    }
)
_VALUE_REASONS: frozenset[DecisionValueMissingReason] = frozenset(
    {
        "domain_inapplicable",
        "value_not_persisted",
        "adapter_not_provided",
        "adapter_returned_no_data",
        "outside_detection_window",
        "no_live_classified_track",
        "bed_region_unavailable",
        "bed_observation_missing",
        "track_no_longer_live",
        "no_observed_person",
    }
)
_TRACK_BED_REASONS: frozenset[TrackBedMissingReason] = frozenset(
    {
        "domain_inapplicable",
        "track_not_persisted",
        "bed_not_persisted",
        "no_live_classified_track",
        "bed_region_unavailable",
        "bed_observation_missing",
        "track_no_longer_live",
        "no_observed_person",
    }
)
_NEIGHBORHOOD_REASONS: dict[CoverageReason, NeighborhoodMissingReason] = {
    "NEIGHBORHOOD_PRUNED": "retention_loss",
    "NEIGHBORHOOD_EPOCH_PREFIX_SHORT": "prefix_shorter_than_window",
    "NEIGHBORHOOD_GAP_UNEXPLAINED": "sequence_gap",
    "NEIGHBORHOOD_CROSSES_BOOT_OR_EPOCH": "neighborhood_pruned",
    "ANALYSIS_TRACE_NOT_RECORDED": "neighborhood_pruned",
    "ANALYSIS_TRACE_DELETED_OR_UNLINKED": "neighborhood_pruned",
    "DECISION_TRACE_NOT_RECORDED": "neighborhood_pruned",
}


class EventExplanationNotFound(Exception):
    """Typed service outcome when no evidence event exists."""

    def __init__(self, edge_event_id: str) -> None:
        self.edge_event_id = edge_event_id
        super().__init__(edge_event_id)


class EventExplanationService:
    """Compose decision-provenance completeness without hiding nested gaps."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def explain(self, edge_event_id: str) -> EventExplanationResponse:
        facts = EventExplanationQuery(self.database_path).get(edge_event_id)
        if facts is None:
            raise EventExplanationNotFound(edge_event_id)
        if isinstance(facts, TraceRefConflict):
            identity = _identity_for_conflict(self.database_path, facts.edge_event_id)
            return _compose(
                database_path=self.database_path,
                identity=identity,
                decision=None,
                runtime=None,
                review=EventExplanationQuery(self.database_path).current_review(
                    facts.edge_event_id
                ),
                provenance="UNAVAILABLE",
                provenance_reasons=("trace_ref_conflict",),
                field_reason="trace_ref_conflict",
                analysis_reason="trace_ref_conflict",
            )
        if facts.decision is None:
            return _compose(
                database_path=self.database_path,
                identity=facts.identity,
                decision=None,
                runtime=None,
                review=facts.review,
                provenance="UNAVAILABLE",
                provenance_reasons=("decision_trace_unresolved",),
                field_reason="decision_trace_unresolved",
                analysis_reason="analysis_trace_unresolved",
            )
        reasons: list[DecisionProvenanceReason] = []
        if facts.runtime is None:
            reasons.append("analysis_trace_unresolved")
        projection = _project_manifest(self.database_path, facts.decision)
        if not _manifest_group_resolved(projection):
            reasons.append("runtime_manifest_unresolved")
        if reasons:
            return _compose(
                database_path=self.database_path,
                identity=facts.identity,
                decision=facts.decision,
                runtime=facts.runtime,
                review=facts.review,
                provenance="PARTIAL",
                provenance_reasons=tuple(reasons),
                field_reason="decision_trace_unresolved",
                analysis_reason="analysis_trace_unresolved",
                projection=projection,
            )
        return _compose(
            database_path=self.database_path,
            identity=facts.identity,
            decision=facts.decision,
            runtime=facts.runtime,
            review=facts.review,
            provenance="COMPLETE",
            provenance_reasons=(),
            field_reason="decision_trace_unresolved",
            analysis_reason="analysis_trace_unresolved",
            projection=projection,
        )


def _compose(
    *,
    database_path: Path,
    identity: EventExplanationIdentity,
    decision: EventExplanationDecision | None,
    runtime: EventExplanationRuntime | None,
    review: EventExplanationCurrentReview | None,
    provenance: DecisionProvenance,
    provenance_reasons: tuple[DecisionProvenanceReason, ...],
    field_reason: DecisionTraceMissingReason,
    analysis_reason: AnalysisMissingReason,
    projection: RuntimeManifestProjection | None = None,
) -> EventExplanationResponse:
    sections = project_explanation_sections(database_path, identity.edge_event_id)
    if projection is None:
        projection = project_runtime_manifest(
            canonical_json=None,
            runtime_manifest_sha256=None,
            module_qualified_id=None,
        )
    neighborhood = _neighborhood(database_path, decision)
    domain, event_type = _domain_and_type(identity.event_type)
    camera_id = identity.camera_id or (None if runtime is None else runtime.camera_id)
    if camera_id is None:
        raise EventExplanationNotFound(identity.edge_event_id)
    probability, threshold, values, missing = _decision_values(decision, domain)
    return EventExplanationResponse(
        decision_provenance=provenance,
        decision_provenance_reasons=list(provenance_reasons),
        edge_event_id=identity.edge_event_id,
        facility_id=FacilityIdFact(missing_reason="facility_id_not_a_first_class_column"),
        camera_id=camera_id,
        domain=domain,
        event_type=event_type,
        detected_at=identity.detected_at,
        worker_boot_id=_optional_text(
            WorkerBootIdFact,
            None if runtime is None else runtime.worker_boot_id,
            analysis_reason,
        ),
        stream_epoch=_optional_int(
            StreamEpochFact,
            None if runtime is None else runtime.stream_epoch,
            analysis_reason,
        ),
        frame_seq=_optional_int(
            FrameSeqFact,
            None if runtime is None else runtime.frame_seq,
            analysis_reason,
        ),
        decision_trace_id=_optional_text(
            DecisionTraceIdFact,
            None if decision is None else decision.decision_trace_id,
            field_reason,
        ),
        reason=_decision_reason(decision, field_reason),
        previous_state=_decision_state(
            None if decision is None else decision.previous_state,
            field_reason,
        ),
        current_state=_decision_state(
            None if decision is None else decision.current_state,
            field_reason,
        ),
        triggered=_optional_bool(
            TriggeredFact,
            None if decision is None else decision.triggered,
            field_reason,
        ),
        probability=probability,
        threshold=threshold,
        decision_values=values,
        missing_values=missing,
        track_id=_track_or_bed(
            TrackIdFact,
            None if decision is None else decision.track_id,
            None if decision is None else decision.track_missing_reason,
            "track_not_persisted",
        ),
        bed_id=_track_or_bed(
            BedIdFact,
            None if decision is None else decision.bed_id,
            None if decision is None else decision.bed_missing_reason,
            "bed_not_persisted",
        ),
        config_version=ConfigVersionFact(
            **_runtime_int_payload(
                projection.config_version,
                projection.config_version_missing_reason,
            )
        ),
        policy_qualified_id=_policy_fact(decision, projection),
        model=ModelFact(
            **_runtime_text_payload(
                projection.model_version,
                projection.model_version_missing_reason,
            )
        ),
        detector_version=DetectorVersionFact(
            **_runtime_text_payload(
                projection.detector_version,
                projection.detector_version_missing_reason,
            )
        ),
        runtime_manifest_sha256=RuntimeManifestSha256Fact(
            **_runtime_text_payload(
                projection.runtime_manifest_sha256,
                projection.runtime_manifest_sha256_missing_reason,
            )
        ),
        worker_build_revision=SourceRevisionFact(
            **_runtime_text_payload(
                projection.worker_build_revision,
                projection.worker_build_revision_missing_reason,
            )
        ),
        image_revision=SourceRevisionFact(
            **_runtime_text_payload(
                projection.image_revision,
                projection.image_revision_missing_reason,
            )
        ),
        delivery=sections.delivery,
        media=sections.media,
        review=project_current_review(review),
        neighborhood=neighborhood,
        correlation=sections.correlation,
    )


def _project_manifest(
    database_path: Path,
    decision: EventExplanationDecision,
) -> RuntimeManifestProjection:
    return project_runtime_manifest(
        canonical_json=_canonical_json(database_path, decision.runtime_manifest_sha256),
        runtime_manifest_sha256=decision.runtime_manifest_sha256,
        module_qualified_id=decision.module_qualified_id,
    )


def _manifest_group_resolved(projection: RuntimeManifestProjection) -> bool:
    reasons = {
        projection.config_version_missing_reason,
        projection.policy_version_missing_reason,
        projection.model_version_missing_reason,
        projection.detector_version_missing_reason,
        projection.worker_build_revision_missing_reason,
        projection.image_revision_missing_reason,
    }
    return not reasons.intersection(
        {
            RuntimeManifestMissingReason.LEGACY_MANIFEST,
            RuntimeManifestMissingReason.MANIFEST_UNAVAILABLE,
            RuntimeManifestMissingReason.MANIFEST_MALFORMED,
        }
    )


def _canonical_json(database_path: Path, manifest_sha: str | None) -> str | None:
    if manifest_sha is None:
        return None
    connection = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        row = connection.execute(
            "SELECT canonical_json FROM runtime_manifest_contents WHERE manifest_sha256 = ?",
            (manifest_sha,),
        ).fetchone()
    finally:
        connection.close()
    if row is None or not isinstance(row[0], str):
        return None
    return row[0]


def _identity_for_conflict(database_path: Path, edge_event_id: str) -> EventExplanationIdentity:
    connection = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        row = connection.execute(
            """
            SELECT event.edge_event_id, incident.incident_id, incident.camera_id,
                   incident.event_type, event.detected_at
            FROM evidence_events AS event
            LEFT JOIN evidence_incidents AS incident
              ON incident.edge_event_id = event.edge_event_id
            WHERE event.edge_event_id = ?
            """,
            (edge_event_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise EventExplanationNotFound(edge_event_id)
    return EventExplanationIdentity(
        edge_event_id=str(row[0]),
        incident_id=None if row[1] is None else str(row[1]),
        camera_id=None if row[2] is None else str(row[2]),
        event_type=None if row[3] is None else str(row[3]),
        detected_at=str(row[4]),
    )


def _neighborhood(
    database_path: Path,
    decision: EventExplanationDecision | None,
) -> EventExplanationNeighborhood:
    if decision is None:
        return EventExplanationNeighborhood(
            status="UNAVAILABLE",
            reasons=["neighborhood_pruned"],
            neighborhood_pruned=True,
            retained_frame_count=0,
        )
    coverage = EventNeighborhoodQuery(database_path).coverage_for_decision(
        decision.decision_trace_id
    )
    return _neighborhood_from_coverage(coverage)


def _neighborhood_from_coverage(coverage: NeighborhoodCoverage) -> EventExplanationNeighborhood:
    if coverage.status == "COMPLETE":
        return EventExplanationNeighborhood(
            status="COMPLETE",
            reasons=[],
            neighborhood_pruned=False,
            retained_frame_count=coverage.retained_frames,
        )
    reason: NeighborhoodMissingReason = "neighborhood_pruned"
    if coverage.coverage_reason is not None:
        reason = _NEIGHBORHOOD_REASONS[coverage.coverage_reason]
    status = "UNAVAILABLE" if coverage.status == "MISSING_TRIGGER" else "PARTIAL"
    return EventExplanationNeighborhood(
        status=status,
        reasons=[reason],
        neighborhood_pruned=True,
        retained_frame_count=coverage.retained_frames,
    )


def _decision_values(
    decision: EventExplanationDecision | None,
    domain: EventDomain,
) -> tuple[
    ProbabilityFact,
    ThresholdFact,
    list[EventExplanationDecisionValue],
    list[EventExplanationMissingValue],
]:
    by_name: dict[DecisionValueName, tuple[float | None, str | None]] = {}
    if decision is not None:
        for item in decision.values:
            if _is_value_name(item.name):
                by_name[item.name] = (item.numeric_value, item.missing_reason)
    values: list[EventExplanationDecisionValue] = []
    missing: list[EventExplanationMissingValue] = []
    for name, (numeric, stored_reason) in by_name.items():
        if numeric is not None:
            values.append(EventExplanationDecisionValue(name=name, value=numeric))
        else:
            missing.append(
                EventExplanationMissingValue(
                    name=name,
                    missing_reason=_value_reason(stored_reason, domain),
                )
            )
    return (
        _named_scalar(ProbabilityFact, by_name.get("fall_probability"), domain),
        _named_scalar(ThresholdFact, by_name.get("operating_threshold"), domain),
        values,
        missing,
    )


@overload
def _named_scalar(
    fact_cls: type[ProbabilityFact],
    stored: tuple[float | None, str | None] | None,
    domain: EventDomain,
) -> ProbabilityFact: ...


@overload
def _named_scalar(
    fact_cls: type[ThresholdFact],
    stored: tuple[float | None, str | None] | None,
    domain: EventDomain,
) -> ThresholdFact: ...


def _named_scalar(
    fact_cls: type[ProbabilityFact] | type[ThresholdFact],
    stored: tuple[float | None, str | None] | None,
    domain: EventDomain,
) -> ProbabilityFact | ThresholdFact:
    if stored is None:
        return fact_cls(missing_reason=_value_reason(None, domain))
    numeric, stored_reason = stored
    if numeric is not None:
        return fact_cls(value=numeric)
    return fact_cls(missing_reason=_value_reason(stored_reason, domain))


def _value_reason(
    stored: str | None,
    domain: EventDomain,
) -> DecisionValueMissingReason:
    mapped = _map_reason(stored)
    if _is_value_reason(mapped):
        return mapped
    if domain == "other":
        return "domain_inapplicable"
    return "value_not_persisted"


@overload
def _track_or_bed(
    fact_cls: type[TrackIdFact],
    value: int | None,
    stored_reason: str | None,
    absent_reason: TrackBedMissingReason,
) -> TrackIdFact: ...


@overload
def _track_or_bed(
    fact_cls: type[BedIdFact],
    value: int | None,
    stored_reason: str | None,
    absent_reason: TrackBedMissingReason,
) -> BedIdFact: ...


def _track_or_bed(
    fact_cls: type[TrackIdFact] | type[BedIdFact],
    value: int | None,
    stored_reason: str | None,
    absent_reason: TrackBedMissingReason,
) -> TrackIdFact | BedIdFact:
    if value is not None:
        return fact_cls(value=value)
    mapped = _map_reason(stored_reason)
    if _is_track_bed_reason(mapped):
        return fact_cls(missing_reason=mapped)
    return fact_cls(missing_reason=absent_reason)


def _map_reason(stored: str | None) -> str | None:
    if stored is None:
        return None
    if stored == "not-applicable":
        return "domain_inapplicable"
    return stored.replace("-", "_")


def _is_value_name(value: str) -> TypeGuard[DecisionValueName]:
    return value in _VALUE_NAMES


def _is_value_reason(value: str | None) -> TypeGuard[DecisionValueMissingReason]:
    return value in _VALUE_REASONS


def _is_track_bed_reason(value: str | None) -> TypeGuard[TrackBedMissingReason]:
    return value in _TRACK_BED_REASONS


def _is_decision_reason(value: str) -> TypeGuard[DecisionReason]:
    return value in _DECISION_REASONS


def _is_decision_state(value: str | None) -> TypeGuard[DecisionState]:
    return value in _DECISION_STATES


def _is_review_disposition(value: str) -> TypeGuard[ReviewDispositionToken]:
    return value in _REVIEW_DISPOSITIONS


def _decision_reason(
    decision: EventExplanationDecision | None,
    field_reason: DecisionTraceMissingReason,
) -> DecisionReasonFact:
    if decision is not None and _is_decision_reason(decision.reason):
        return DecisionReasonFact(value=decision.reason)
    return DecisionReasonFact(missing_reason=field_reason)


def _decision_state(
    value: str | None,
    field_reason: DecisionTraceMissingReason,
) -> DecisionStateFact:
    if _is_decision_state(value):
        return DecisionStateFact(value=value)
    return DecisionStateFact(missing_reason=field_reason)


def _policy_fact(
    decision: EventExplanationDecision | None,
    projection: RuntimeManifestProjection,
) -> PolicyQualifiedIdFact:
    if decision is not None:
        try:
            return PolicyQualifiedIdFact(value=decision.policy_qualified_id)
        except ValidationError:
            return PolicyQualifiedIdFact(missing_reason="persisted_value_invalid")
    return PolicyQualifiedIdFact(
        **_runtime_text_payload(
            projection.policy_version,
            projection.policy_version_missing_reason,
        )
    )


class _RuntimeIntPayload(TypedDict):
    value: int | None
    missing_reason: RuntimeMissingReason | None


class _RuntimeTextPayload(TypedDict):
    value: str | None
    missing_reason: RuntimeMissingReason | None


def _runtime_int_payload(
    value: int | None,
    reason: RuntimeManifestMissingReason | None,
) -> _RuntimeIntPayload:
    if value is not None:
        return {"value": value, "missing_reason": None}
    return {"value": None, "missing_reason": _runtime_reason(reason)}


def _runtime_text_payload(
    value: str | None,
    reason: RuntimeManifestMissingReason | None,
) -> _RuntimeTextPayload:
    if value is not None:
        return {"value": value, "missing_reason": None}
    return {"value": None, "missing_reason": _runtime_reason(reason)}


def _runtime_reason(
    reason: RuntimeManifestMissingReason | None,
) -> RuntimeMissingReason:
    if reason is RuntimeManifestMissingReason.LEGACY_MANIFEST:
        return "legacy_manifest_field"
    if reason is RuntimeManifestMissingReason.FIELD_UNAVAILABLE:
        return "field_not_persisted"
    if reason is RuntimeManifestMissingReason.FIELD_MALFORMED:
        return "field_not_persisted"
    return "runtime_manifest_unresolved"


def _optional_text(
    fact_cls: type,
    value: str | None,
    missing_reason: AnalysisMissingReason | DecisionTraceMissingReason,
):
    if value is not None:
        return fact_cls(value=value)
    return fact_cls(missing_reason=missing_reason)


def _optional_int(
    fact_cls: type,
    value: int | None,
    missing_reason: AnalysisMissingReason,
):
    if value is not None:
        return fact_cls(value=value)
    return fact_cls(missing_reason=missing_reason)


def _optional_bool(
    fact_cls: type,
    value: bool | None,
    missing_reason: DecisionTraceMissingReason,
):
    if value is not None:
        return fact_cls(value=value)
    return fact_cls(missing_reason=missing_reason)


def _domain_and_type(event_type: str | None) -> tuple[EventDomain, EventType]:
    if event_type == "fall":
        return "fall", "fall"
    if event_type == "bed-exit":
        return "bed-exit", "bed-exit"
    return "other", "other"


def project_current_review(
    review: EventExplanationCurrentReview | None,
) -> EventExplanationReview:
    if review is None:
        return EventExplanationReview(
            status="UNAVAILABLE",
            reasons=["review_not_recorded"],
            disposition=ReviewDispositionFact(missing_reason="review_not_recorded"),
        )
    if not _is_review_disposition(review.disposition):
        return EventExplanationReview(
            status="UNAVAILABLE",
            reasons=["persisted_value_invalid"],
            disposition=ReviewDispositionFact(missing_reason="persisted_value_invalid"),
        )
    return EventExplanationReview(
        status="COMPLETE",
        reasons=[],
        disposition=ReviewDispositionFact(value=review.disposition),
    )


__all__ = [
    "EventExplanationNotFound",
    "EventExplanationService",
    "project_current_review",
]
