"""Standalone read-only production false-positive attribution package."""

from worker.fp_attribution.attribution import (
    PREDICATE_REGISTRY,
    AttributionAnnotations,
    AttributionCategory,
    AttributionDecision,
    PredicateSpec,
    classify_record,
    machine_bytes,
)
from worker.fp_attribution.cohort import (
    FalsePositiveCohort,
    FalsePositiveCohortExclusion,
    FalsePositiveCohortMember,
    FalsePositiveCohortQuery,
    open_query_only_connection,
)
from worker.fp_attribution.evidence import (
    AttributionEvidence,
    AttributionEvidenceQuery,
    AttributionEvidenceRecord,
)
from worker.fp_attribution.metrics import (
    AlertIdMetric,
    AttributionMetricEvent,
    AttributionMetrics,
    CategoryShare,
    MetricRatio,
    TransportMetrics,
    metric_event_from_record,
    metrics_machine_bytes,
    summarize_attribution_metrics,
)

__all__ = [
    "AlertIdMetric",
    "AttributionAnnotations",
    "AttributionCategory",
    "AttributionDecision",
    "AttributionEvidence",
    "AttributionEvidenceQuery",
    "AttributionEvidenceRecord",
    "AttributionMetricEvent",
    "AttributionMetrics",
    "CategoryShare",
    "FalsePositiveCohort",
    "FalsePositiveCohortExclusion",
    "FalsePositiveCohortMember",
    "FalsePositiveCohortQuery",
    "MetricRatio",
    "PREDICATE_REGISTRY",
    "PredicateSpec",
    "TransportMetrics",
    "classify_record",
    "machine_bytes",
    "metric_event_from_record",
    "metrics_machine_bytes",
    "open_query_only_connection",
    "summarize_attribution_metrics",
]
