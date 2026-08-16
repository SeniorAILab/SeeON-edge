"""Strict Event Explanation response contracts.

Privacy-bounded Pydantic v2 models for GET /events/{edge_event_id}/explanation.
This module defines the wire contract only: no worker import and no payload parsing.
"""

from __future__ import annotations

import re
from typing import ClassVar, Generic, Literal, Self, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DecisionProvenance: TypeAlias = Literal["COMPLETE", "PARTIAL", "UNAVAILABLE"]
SectionStatus: TypeAlias = Literal["COMPLETE", "PARTIAL", "UNAVAILABLE"]
EventDomain: TypeAlias = Literal["fall", "bed-exit", "other"]
EventType: TypeAlias = Literal["fall", "bed-exit", "other"]

DecisionReason: TypeAlias = Literal[
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
]
DecisionState: TypeAlias = Literal[
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
]
DecisionValueName: TypeAlias = Literal[
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
]
DecisionValueMissingReason: TypeAlias = Literal[
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
]
FacilityMissingReason: TypeAlias = Literal["facility_id_not_a_first_class_column"]
AnalysisMissingReason: TypeAlias = Literal[
    "analysis_trace_unresolved",
    "trace_ref_conflict",
]
DecisionTraceMissingReason: TypeAlias = Literal[
    "decision_trace_unresolved",
    "trace_ref_conflict",
]
TrackBedMissingReason: TypeAlias = Literal[
    "domain_inapplicable",
    "track_not_persisted",
    "bed_not_persisted",
    "no_live_classified_track",
    "bed_region_unavailable",
    "bed_observation_missing",
    "track_no_longer_live",
    "no_observed_person",
]
RuntimeMissingReason: TypeAlias = Literal[
    "runtime_manifest_unresolved",
    "legacy_manifest_field",
    "field_not_persisted",
    "persisted_value_invalid",
]
OutboxState: TypeAlias = Literal["PENDING", "ACKED", "PERMANENT", "COMPATIBILITY"]
DeliveryDispositionToken: TypeAlias = Literal["RETRY", "PERMANENT", "COMPATIBILITY"]
DeliveryMissingReason: TypeAlias = Literal[
    "disposition_not_persisted",
    "last_http_status_not_persisted",
    "backend_event_id_not_persisted",
    "delivery_never_attempted",
    "outbox_row_unresolved",
]
ArtifactState: TypeAlias = Literal[
    "PENDING",
    "AVAILABLE",
    "UNAVAILABLE",
    "CORRUPT",
    "NOT_RECORDED",
]
MediaMissingReason: TypeAlias = Literal[
    "snapshot_not_recorded",
    "clip_not_recorded",
    "artifact_unavailable",
]
ReviewDispositionToken: TypeAlias = Literal["TRUE_POSITIVE", "FALSE_POSITIVE"]
ReviewMissingReason: TypeAlias = Literal["review_not_recorded"]
NeighborhoodMissingReason: TypeAlias = Literal[
    "neighborhood_pruned",
    "prefix_shorter_than_window",
    "sequence_gap",
    "retention_loss",
]
CorrelationMissingReason: TypeAlias = Literal["alert_correlation_export_not_supplied"]
DecisionProvenanceReason: TypeAlias = Literal[
    "decision_trace_unresolved",
    "trace_ref_conflict",
    "analysis_trace_unresolved",
    "runtime_manifest_unresolved",
    "required_decision_group_missing",
]


class _ExplanationModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class _ValueReasonPair(_ExplanationModel):
    @model_validator(mode="after")
    def require_value_xor_missing_reason(self) -> Self:
        has_value = self.value is not None
        has_reason = self.missing_reason is not None
        if has_value == has_reason:
            raise ValueError("value and missing_reason must be mutually exclusive")
        return self


class FacilityIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: FacilityMissingReason | None = None


class WorkerBootIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: AnalysisMissingReason | None = None


class StreamEpochFact(_ValueReasonPair):
    value: int | None = Field(default=None, ge=0)
    missing_reason: AnalysisMissingReason | None = None


class FrameSeqFact(_ValueReasonPair):
    value: int | None = Field(default=None, ge=0)
    missing_reason: AnalysisMissingReason | None = None


class DecisionTraceIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=64, max_length=64)
    missing_reason: DecisionTraceMissingReason | None = None


class DecisionReasonFact(_ValueReasonPair):
    value: DecisionReason | None = None
    missing_reason: DecisionTraceMissingReason | None = None


class DecisionStateFact(_ValueReasonPair):
    value: DecisionState | None = None
    missing_reason: DecisionTraceMissingReason | None = None


class TriggeredFact(_ValueReasonPair):
    value: bool | None = None
    missing_reason: DecisionTraceMissingReason | None = None


class ProbabilityFact(_ValueReasonPair):
    value: float | None = None
    missing_reason: DecisionValueMissingReason | None = None


class ThresholdFact(_ValueReasonPair):
    value: float | None = None
    missing_reason: DecisionValueMissingReason | None = None


class TrackIdFact(_ValueReasonPair):
    value: int | None = None
    missing_reason: TrackBedMissingReason | None = None


class BedIdFact(_ValueReasonPair):
    value: int | None = None
    missing_reason: TrackBedMissingReason | None = None


class ConfigVersionFact(_ValueReasonPair):
    value: int | None = Field(default=None, ge=0)
    missing_reason: RuntimeMissingReason | None = None


class PolicyQualifiedIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1, max_length=64)
    missing_reason: RuntimeMissingReason | None = None

    @field_validator("value")
    @classmethod
    def require_versioned_policy_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _is_versioned_policy_identity(value):
            raise ValueError("policy_qualified_id is not a versioned policy identity")
        return value


class ModelFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: RuntimeMissingReason | None = None


class DetectorVersionFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: RuntimeMissingReason | None = None


class RuntimeManifestSha256Fact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    missing_reason: RuntimeMissingReason | None = None


class SourceRevisionFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    missing_reason: RuntimeMissingReason | None = None


class OutboxStateFact(_ValueReasonPair):
    value: OutboxState | None = None
    missing_reason: DeliveryMissingReason | None = None


class AttemptCountFact(_ValueReasonPair):
    value: int | None = Field(default=None, ge=0)
    missing_reason: DeliveryMissingReason | None = None


class DeliveryDispositionFact(_ValueReasonPair):
    value: DeliveryDispositionToken | None = None
    missing_reason: DeliveryMissingReason | None = None


class HttpStatusFact(_ValueReasonPair):
    value: int | None = Field(default=None, ge=100, le=599)
    missing_reason: DeliveryMissingReason | None = None


class BackendEventIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: DeliveryMissingReason | None = None


class ReviewDispositionFact(_ValueReasonPair):
    value: ReviewDispositionToken | None = None
    missing_reason: ReviewMissingReason | None = None


class AlertIdFact(_ValueReasonPair):
    value: str | None = Field(default=None, min_length=1)
    missing_reason: CorrelationMissingReason | None = None


class EventExplanationDecisionValue(_ExplanationModel):
    name: DecisionValueName
    value: float


class EventExplanationMissingValue(_ExplanationModel):
    name: DecisionValueName
    missing_reason: DecisionValueMissingReason


ReasonT = TypeVar("ReasonT", bound=str)


class EventExplanationSection(_ExplanationModel, Generic[ReasonT]):
    status: SectionStatus
    reasons: list[ReasonT]

    @model_validator(mode="after")
    def require_closed_section_reasons(self) -> Self:
        if self.status == "COMPLETE" and self.reasons:
            raise ValueError("COMPLETE sections cannot carry reasons")
        if self.status != "COMPLETE" and not self.reasons:
            raise ValueError("PARTIAL and UNAVAILABLE sections require reasons")
        return self


class EventExplanationDelivery(EventExplanationSection[DeliveryMissingReason]):
    outbox_state: OutboxStateFact
    attempt_count: AttemptCountFact
    last_delivery_disposition: DeliveryDispositionFact
    last_http_status: HttpStatusFact
    backend_event_id: BackendEventIdFact


class EventExplanationArtifact(_ExplanationModel):
    state: ArtifactState
    missing_reason: MediaMissingReason | None = None

    @model_validator(mode="after")
    def require_state_xor_reason(self) -> Self:
        needs_reason = self.state in {"UNAVAILABLE", "CORRUPT", "NOT_RECORDED"}
        has_reason = self.missing_reason is not None
        if needs_reason != has_reason:
            raise ValueError("artifact missing_reason must accompany unavailable states")
        return self


class EventExplanationMedia(EventExplanationSection[MediaMissingReason]):
    snapshot: EventExplanationArtifact
    clip: EventExplanationArtifact


class EventExplanationReview(EventExplanationSection[ReviewMissingReason]):
    disposition: ReviewDispositionFact


class EventExplanationNeighborhood(EventExplanationSection[NeighborhoodMissingReason]):
    neighborhood_pruned: bool
    retained_frame_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_window(self) -> Self:
        if self.status == "COMPLETE":
            if self.neighborhood_pruned or self.retained_frame_count != 30:
                raise ValueError("COMPLETE neighborhood requires an exact 30-frame window")
        elif not self.neighborhood_pruned:
            raise ValueError("incomplete neighborhood must be pruned")
        return self


class EventExplanationCorrelation(EventExplanationSection[CorrelationMissingReason]):
    alert_id: AlertIdFact


class EventExplanationResponse(_ExplanationModel):
    decision_provenance: DecisionProvenance
    decision_provenance_reasons: list[DecisionProvenanceReason]
    edge_event_id: str = Field(min_length=1)
    facility_id: FacilityIdFact
    camera_id: str = Field(min_length=1)
    domain: EventDomain
    event_type: EventType
    detected_at: str = Field(min_length=1)
    worker_boot_id: WorkerBootIdFact
    stream_epoch: StreamEpochFact
    frame_seq: FrameSeqFact
    decision_trace_id: DecisionTraceIdFact
    reason: DecisionReasonFact
    previous_state: DecisionStateFact
    current_state: DecisionStateFact
    triggered: TriggeredFact
    probability: ProbabilityFact
    threshold: ThresholdFact
    decision_values: list[EventExplanationDecisionValue]
    missing_values: list[EventExplanationMissingValue]
    track_id: TrackIdFact
    bed_id: BedIdFact
    config_version: ConfigVersionFact
    policy_qualified_id: PolicyQualifiedIdFact
    model: ModelFact
    detector_version: DetectorVersionFact
    runtime_manifest_sha256: RuntimeManifestSha256Fact
    worker_build_revision: SourceRevisionFact
    image_revision: SourceRevisionFact
    delivery: EventExplanationDelivery
    media: EventExplanationMedia
    review: EventExplanationReview
    neighborhood: EventExplanationNeighborhood
    correlation: EventExplanationCorrelation

    @model_validator(mode="after")
    def require_provenance_reasons(self) -> Self:
        if self.decision_provenance == "COMPLETE" and self.decision_provenance_reasons:
            raise ValueError("COMPLETE decision provenance cannot carry reasons")
        if self.decision_provenance != "COMPLETE" and not self.decision_provenance_reasons:
            raise ValueError("PARTIAL and UNAVAILABLE decision provenance require reasons")
        names = [item.name for item in (*self.decision_values, *self.missing_values)]
        if len(names) != len(set(names)):
            raise ValueError("decision values cannot be both known and missing")
        if (
            self.decision_provenance == "COMPLETE"
            and self.facility_id.missing_reason != "facility_id_not_a_first_class_column"
        ):
            raise ValueError("current schema facility_id must be explicitly unavailable")
        return self


_VERSIONED_POLICY_IDENTITY = re.compile(
    r"\A[a-z][a-z0-9_]{0,31}\.policy\.v[1-9][0-9]{0,2}\Z"
)


def _is_versioned_policy_identity(value: str) -> bool:
    return _VERSIONED_POLICY_IDENTITY.fullmatch(value) is not None


__all__ = [
    "EventExplanationResponse",
    "PolicyQualifiedIdFact",
]
