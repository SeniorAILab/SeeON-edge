"""Exact attribution and transport metrics over Todo 10-12 outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from worker.fp_attribution.attribution import AttributionDecision, classify_record
from worker.fp_attribution.cohort import FalsePositiveCohortExclusion
from worker.fp_attribution.evidence import AttributionEvidenceRecord

_ALERT_EXPORT_MISSING = "alert_correlation_export_not_supplied"
_CATEGORY_VOCABULARY = (
    "BACKEND_OR_UI_DUPLICATE",
    "DELIVERY_RETRY",
    "BED_STALE_TRACK",
    "FALL_LATCH_REARM",
    "EPISODE_FRAGMENTATION",
    "TRACKER_OR_IDENTITY",
    "ZERO_OR_MISSING_POSE",
    "BED_GEOMETRY_OR_ASSIGNMENT",
    "CAMERA_LIGHTING_OR_DECODE",
    "INSUFFICIENT_EVIDENCE",
    "TRANSPORT_ONLY",
    "UNCATEGORIZED",
)
_ALERT_KEYS = frozenset({"edge_event_id", "alert_id"})
AlertAvailability = Literal["AVAILABLE", "UNAVAILABLE"]


@dataclass(frozen=True, slots=True)
class MetricRatio:
    value: float | None
    numerator: int
    denominator: int | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class CategoryShare:
    category: str
    count: int
    ratio: MetricRatio


@dataclass(frozen=True, slots=True)
class AlertIdMetric:
    status: AlertAvailability
    value: int | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class TransportMetrics:
    unique_edge_event_count: int
    total_attempts: int
    extra_attempts_beyond_first: int
    backend_event_id_available_count: int
    distinct_backend_event_id_count: int
    proof_backed_duplicate_count: int
    proof_backed_retry_count: int
    unique_alert_id: AlertIdMetric


@dataclass(frozen=True, slots=True)
class AttributionMetricEvent:
    edge_event_id: str
    evidence_status: str
    category: str | None
    neighborhood_pruned: bool
    attempt_count: int
    backend_event_ids: tuple[str, ...]
    correlation_status: str
    correlation_kind: str | None


@dataclass(frozen=True, slots=True)
class AttributionMetrics:
    cohort_total: int
    attributable_count: int
    pruned_count: int
    unknown_count: int
    legacy_excluded_count: int
    legacy_excluded_census: dict[str, int]
    attribution_rate: MetricRatio
    retention_coverage: MetricRatio
    attribution_coverage: MetricRatio
    category_counts: tuple[CategoryShare, ...]
    transport: TransportMetrics


def metric_event_from_record(
    record: AttributionEvidenceRecord,
    *,
    correlation_export: object | None = None,
    decision: AttributionDecision | None = None,
) -> AttributionMetricEvent:
    resolved = (
        decision
        if decision is not None
        else classify_record(record, correlation_export=correlation_export)
    )
    return AttributionMetricEvent(
        edge_event_id=record.edge_event_id,
        evidence_status=resolved.evidence_status,
        category=resolved.category,
        neighborhood_pruned=resolved.annotations.neighborhood_pruned,
        attempt_count=resolved.annotations.attempt_count,
        backend_event_ids=resolved.annotations.backend_event_ids,
        correlation_status=resolved.annotations.correlation_status,
        correlation_kind=resolved.annotations.correlation_kind,
    )


def summarize_attribution_metrics(
    events: Sequence[
        AttributionEvidenceRecord | AttributionDecision | AttributionMetricEvent
    ],
    *,
    exclusions: Sequence[FalsePositiveCohortExclusion] | None = None,
    alert_correlation_export: object | None = None,
) -> AttributionMetrics:
    rows = tuple(_normalize_event(item) for item in events)
    _reject_duplicate_event_ids(rows)
    attributable, pruned, unknown = _partition(rows)
    if attributable + pruned + unknown != len(rows):
        raise ValueError("partition_inconsistent")
    category_counts = _category_shares(rows, attributable)
    census = _exclusion_census(exclusions)
    return AttributionMetrics(
        cohort_total=len(rows),
        attributable_count=attributable,
        pruned_count=pruned,
        unknown_count=unknown,
        legacy_excluded_count=sum(census.values()),
        legacy_excluded_census=census,
        attribution_rate=_ratio(attributable, len(rows), zero_reason="cohort_total_zero"),
        retention_coverage=_ratio(
            unknown + attributable,
            len(rows),
            zero_reason="cohort_total_zero",
        ),
        attribution_coverage=_ratio(
            attributable,
            unknown + attributable,
            zero_reason="evaluable_total_zero",
        ),
        category_counts=category_counts,
        transport=_transport(rows, alert_correlation_export),
    )


def metrics_machine_bytes(summary: AttributionMetrics) -> bytes:
    return json.dumps(_payload(summary), separators=(",", ":"), sort_keys=True).encode("utf-8")


def _normalize_event(
    event: AttributionEvidenceRecord | AttributionDecision | AttributionMetricEvent,
) -> AttributionMetricEvent:
    if isinstance(event, AttributionMetricEvent):
        return event
    if isinstance(event, AttributionEvidenceRecord):
        return metric_event_from_record(event)
    if isinstance(event, AttributionDecision):
        return AttributionMetricEvent(
            edge_event_id=_require_event_id(event),
            evidence_status=event.evidence_status,
            category=event.category,
            neighborhood_pruned=event.annotations.neighborhood_pruned,
            attempt_count=event.annotations.attempt_count,
            backend_event_ids=event.annotations.backend_event_ids,
            correlation_status=event.annotations.correlation_status,
            correlation_kind=event.annotations.correlation_kind,
        )
    raise TypeError("attribution metric event is invalid")


def _require_event_id(decision: AttributionDecision) -> str:
    event_id = getattr(decision, "edge_event_id", None)
    if isinstance(event_id, str) and event_id:
        return event_id
    raise ValueError("duplicate_edge_event_id")


def _reject_duplicate_event_ids(rows: Sequence[AttributionMetricEvent]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.edge_event_id in seen:
            raise ValueError("duplicate_edge_event_id")
        seen.add(row.edge_event_id)


def _partition(rows: Sequence[AttributionMetricEvent]) -> tuple[int, int, int]:
    attributable = 0
    pruned = 0
    unknown = 0
    for row in rows:
        bucket = _bucket(row)
        if bucket == "attributable":
            attributable += 1
        elif bucket == "pruned":
            pruned += 1
        else:
            unknown += 1
    return attributable, pruned, unknown


def _bucket(row: AttributionMetricEvent) -> Literal["attributable", "pruned", "unknown"]:
    if row.evidence_status == "PRUNED":
        if row.category is not None or not row.neighborhood_pruned:
            raise ValueError("partition_inconsistent")
        return "pruned"
    if row.evidence_status == "UNKNOWN":
        if row.category is not None or row.neighborhood_pruned:
            raise ValueError("partition_inconsistent")
        return "unknown"
    if row.evidence_status == "COMPLETE":
        if row.category is None or row.neighborhood_pruned:
            raise ValueError("partition_inconsistent")
        if row.category not in _CATEGORY_VOCABULARY:
            raise ValueError("category_total_mismatch")
        return "attributable"
    raise ValueError("partition_inconsistent")


def _category_shares(
    rows: Sequence[AttributionMetricEvent],
    attributable: int,
) -> tuple[CategoryShare, ...]:
    counts = dict.fromkeys(_CATEGORY_VOCABULARY, 0)
    observed = 0
    for row in rows:
        if row.category is None:
            continue
        if row.category not in counts:
            raise ValueError("category_total_mismatch")
        counts[row.category] += 1
        observed += 1
    if observed != attributable:
        raise ValueError("category_total_mismatch")
    if attributable == 0:
        return ()
    return tuple(
        CategoryShare(
            category=category,
            count=count,
            ratio=_ratio(count, attributable, zero_reason="attributable_count_zero"),
        )
        for category, count in counts.items()
        if count
    )


def _exclusion_census(
    exclusions: Sequence[FalsePositiveCohortExclusion] | None,
) -> dict[str, int]:
    census: dict[str, int] = {}
    if exclusions is None:
        return census
    for item in exclusions:
        census[item.reason] = census.get(item.reason, 0) + 1
    return dict(sorted(census.items()))


def _transport(
    rows: Sequence[AttributionMetricEvent],
    alert_export: object | None,
) -> TransportMetrics:
    backend_ids: set[str] = set()
    backend_available = 0
    total_attempts = 0
    extra_attempts = 0
    proof_duplicate = 0
    proof_retry = 0
    for row in rows:
        total_attempts += row.attempt_count
        extra_attempts += max(row.attempt_count - 1, 0)
        if row.backend_event_ids:
            backend_available += 1
            backend_ids.update(row.backend_event_ids)
        if _proof_kind(row) == "BACKEND_OR_UI_DUPLICATE":
            proof_duplicate += 1
        elif _proof_kind(row) == "DELIVERY_RETRY":
            proof_retry += 1
    return TransportMetrics(
        unique_edge_event_count=len(rows),
        total_attempts=total_attempts,
        extra_attempts_beyond_first=extra_attempts,
        backend_event_id_available_count=backend_available,
        distinct_backend_event_id_count=len(backend_ids),
        proof_backed_duplicate_count=proof_duplicate,
        proof_backed_retry_count=proof_retry,
        unique_alert_id=_alert_metric(alert_export),
    )


def _proof_kind(row: AttributionMetricEvent) -> str | None:
    if row.correlation_status == "accepted" and row.correlation_kind in {
        "BACKEND_OR_UI_DUPLICATE",
        "DELIVERY_RETRY",
    }:
        return row.correlation_kind
    if row.category in {"BACKEND_OR_UI_DUPLICATE", "DELIVERY_RETRY"}:
        return row.category
    return None


def _alert_metric(export: object | None) -> AlertIdMetric:
    if export is None:
        return _unavailable_alert()
    if export == ():
        return AlertIdMetric(status="AVAILABLE", value=0, missing_reason=None)
    if not isinstance(export, tuple | list):
        return _unavailable_alert()
    alert_ids: set[str] = set()
    for item in export:
        parsed = _parse_alert_row(item)
        if parsed is None:
            return _unavailable_alert()
        alert_ids.add(parsed)
    return AlertIdMetric(status="AVAILABLE", value=len(alert_ids), missing_reason=None)


def _parse_alert_row(item: object) -> str | None:
    if not isinstance(item, Mapping) or set(item) != _ALERT_KEYS:
        return None
    event_id = item.get("edge_event_id")
    alert_id = item.get("alert_id")
    if not isinstance(event_id, str) or not event_id:
        return None
    if not isinstance(alert_id, str) or not alert_id:
        return None
    return alert_id


def _unavailable_alert() -> AlertIdMetric:
    return AlertIdMetric(
        status="UNAVAILABLE",
        value=None,
        missing_reason=_ALERT_EXPORT_MISSING,
    )


def _ratio(numerator: int, denominator: int, *, zero_reason: str) -> MetricRatio:
    if denominator <= 0:
        return MetricRatio(
            value=None,
            numerator=numerator,
            denominator=None,
            missing_reason=zero_reason,
        )
    return MetricRatio(
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        missing_reason=None,
    )


def _payload(summary: AttributionMetrics) -> dict[str, object]:
    return {
        "attributable_count": summary.attributable_count,
        "attribution_coverage": _ratio_payload(summary.attribution_coverage),
        "attribution_rate": _ratio_payload(summary.attribution_rate),
        "category_counts": [
            {
                "category": item.category,
                "count": item.count,
                "ratio": _ratio_payload(item.ratio),
            }
            for item in summary.category_counts
        ],
        "cohort_total": summary.cohort_total,
        "legacy_excluded_census": summary.legacy_excluded_census,
        "legacy_excluded_count": summary.legacy_excluded_count,
        "pruned_count": summary.pruned_count,
        "retention_coverage": _ratio_payload(summary.retention_coverage),
        "transport": {
            "backend_event_id_available_count": (
                summary.transport.backend_event_id_available_count
            ),
            "distinct_backend_event_id_count": summary.transport.distinct_backend_event_id_count,
            "extra_attempts_beyond_first": summary.transport.extra_attempts_beyond_first,
            "proof_backed_duplicate_count": summary.transport.proof_backed_duplicate_count,
            "proof_backed_retry_count": summary.transport.proof_backed_retry_count,
            "total_attempts": summary.transport.total_attempts,
            "unique_alert_id": {
                "missing_reason": summary.transport.unique_alert_id.missing_reason,
                "status": summary.transport.unique_alert_id.status,
                "value": summary.transport.unique_alert_id.value,
            },
            "unique_edge_event_count": summary.transport.unique_edge_event_count,
        },
        "unknown_count": summary.unknown_count,
    }


def _ratio_payload(ratio: MetricRatio) -> dict[str, float | int | str | None]:
    return {
        "denominator": ratio.denominator,
        "missing_reason": ratio.missing_reason,
        "numerator": ratio.numerator,
        "value": ratio.value,
    }


__all__ = [
    "AlertIdMetric",
    "AttributionMetricEvent",
    "AttributionMetrics",
    "CategoryShare",
    "MetricRatio",
    "TransportMetrics",
    "metric_event_from_record",
    "metrics_machine_bytes",
    "summarize_attribution_metrics",
]
