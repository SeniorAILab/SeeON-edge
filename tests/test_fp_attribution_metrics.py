from __future__ import annotations

from test_fp_attribution_precedence import _complete_record, _duplicate_export, _retry_export

from worker.fp_attribution import (
    AttributionDecision,
    AttributionEvidenceRecord,
    AttributionMetricEvent,
    FalsePositiveCohortExclusion,
    classify_record,
    metric_event_from_record,
)

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
_ALERT_EXPORT_MISSING = "alert_correlation_export_not_supplied"


def _summarize(
    events: tuple[object, ...],
    *,
    exclusions: tuple[FalsePositiveCohortExclusion, ...] | None = None,
    alert_correlation_export: object | None = ...,
):
    from worker.fp_attribution import summarize_attribution_metrics

    kwargs: dict[str, object] = {}
    if exclusions is not None:
        kwargs["exclusions"] = exclusions
    if alert_correlation_export is not ...:
        kwargs["alert_correlation_export"] = alert_correlation_export
    return summarize_attribution_metrics(events, **kwargs)


def _ratio(*, value: float | None, numerator: int, denominator: int | None, reason: str | None):
    from worker.fp_attribution import MetricRatio

    return MetricRatio(
        value=value,
        numerator=numerator,
        denominator=denominator,
        missing_reason=reason,
    )


def _unavailable_ratio(*, numerator: int, reason: str):
    return _ratio(value=None, numerator=numerator, denominator=None, reason=reason)


def _defined_ratio(*, numerator: int, denominator: int):
    return _ratio(
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        reason=None,
    )


def _classified(
    record: AttributionEvidenceRecord,
    correlation: object | None = None,
) -> AttributionMetricEvent:
    return metric_event_from_record(
        record,
        decision=classify_record(record, correlation_export=correlation),
    )


def _pruned_record(*, event_id: str = "event:pruned") -> AttributionEvidenceRecord:
    return _complete_record(
        edge_event_id=event_id,
        evidence_status="PRUNED",
        coverage_status="GAP",
        coverage_reason="NEIGHBORHOOD_GAP_UNEXPLAINED",
        retained_frames=29,
        neighborhood_pruned=True,
        prevented_eligible=False,
        decision_reason="stale-track-exit",
        track_changed=True,
        attempt_count=2,
        backend_event_ids=("backend:pruned",),
    )


def _unknown_record(*, event_id: str = "event:unknown") -> AttributionEvidenceRecord:
    return _complete_record(
        edge_event_id=event_id,
        evidence_status="UNKNOWN",
        score=None,
        threshold=None,
        score_missing_reason="value_not_persisted",
        threshold_missing_reason="value_not_persisted",
        prevented_eligible=False,
        decision_reason="stale-track-exit",
        attempt_count=1,
        backend_event_ids=(),
    )


def _stale_record(*, event_id: str = "event:stale") -> AttributionEvidenceRecord:
    return _complete_record(
        edge_event_id=event_id,
        decision_reason="stale-track-exit",
        previous_state="live-grace",
        current_state="triggered",
        attempt_count=1,
        backend_event_ids=("backend:stale",),
    )


def _tracker_record(*, event_id: str = "event:tracker") -> AttributionEvidenceRecord:
    return _complete_record(
        edge_event_id=event_id,
        decision_reason="fall-active",
        previous_state="fall",
        current_state="fall",
        track_changed=True,
        attempt_count=4,
        backend_event_ids=("backend:shared",),
    )


def _uncategorized_record(*, event_id: str = "event:bare") -> AttributionEvidenceRecord:
    return _complete_record(
        edge_event_id=event_id,
        attempt_count=3,
        backend_event_ids=("backend:shared",),
    )


def _mixed_events() -> tuple[AttributionDecision | AttributionEvidenceRecord, ...]:
    return (
        _classified(_stale_record()),
        _classified(_tracker_record()),
        _classified(_uncategorized_record()),
        _pruned_record(),
        _unknown_record(),
    )


def _assert_partition(summary: object, *, attributable: int, pruned: int, unknown: int) -> None:
    total = attributable + pruned + unknown
    assert summary.cohort_total == total
    assert summary.attributable_count == attributable
    assert summary.pruned_count == pruned
    assert summary.unknown_count == unknown
    assert (
        summary.attributable_count + summary.pruned_count + summary.unknown_count
        == summary.cohort_total
    )


def test_todo10_12_result_shapes_remain_the_metric_inputs() -> None:
    """Characterize committed Todo 10-12 outputs before any metric collapse.

    Given one current-FP exclusion census, one COMPLETE/PRUNED/UNKNOWN evidence
    triple, and one classified COMPLETE row
    When those committed shapes are inspected
    Then exclusions stay typed, evidence categories stay null, and classification
    remains a single category plus orthogonal transport annotations.
    """

    exclusions = (
        FalsePositiveCohortExclusion("TRUE_POSITIVE"),
        FalsePositiveCohortExclusion("UNREVIEWED"),
        FalsePositiveCohortExclusion("UNMAPPABLE_LEGACY"),
    )
    complete = _complete_record()
    pruned = _pruned_record()
    unknown = _unknown_record()
    classified = classify_record(
        _complete_record(attempt_count=9, backend_event_ids=("backend:one",))
    )

    assert tuple(item.reason for item in exclusions) == (
        "TRUE_POSITIVE",
        "UNREVIEWED",
        "UNMAPPABLE_LEGACY",
    )
    assert complete.evidence_status == "COMPLETE"
    assert complete.category is None
    assert pruned.evidence_status == "PRUNED"
    assert pruned.category is None
    assert unknown.evidence_status == "UNKNOWN"
    assert unknown.category is None
    assert classified.category == "UNCATEGORIZED"
    assert classified.annotations.attempt_count == 9
    assert classified.annotations.backend_event_ids == ("backend:one",)


def test_mixed_cohort_emits_exact_counts_ratios_and_transport() -> None:
    """A mixed current-FP cohort keeps detection partitions orthogonal to transport.

    Given three attributable events, one pruned, one unknown, and two legacy exclusions
    When summarize_attribution_metrics runs without an alert export
    Then partitions sum to five, category ratios use attributable=3, and alert
    correlation stays typed unavailable.
    """

    summary = _summarize(
        _mixed_events(),
        exclusions=(
            FalsePositiveCohortExclusion("UNMAPPABLE_LEGACY"),
            FalsePositiveCohortExclusion("OPERATOR_ONLY_LEFTOVER"),
        ),
    )

    _assert_partition(summary, attributable=3, pruned=1, unknown=1)
    assert summary.legacy_excluded_count == 2
    assert summary.legacy_excluded_census == {
        "OPERATOR_ONLY_LEFTOVER": 1,
        "UNMAPPABLE_LEGACY": 1,
    }
    assert summary.attribution_rate == _defined_ratio(numerator=3, denominator=5)
    assert summary.retention_coverage == _defined_ratio(numerator=4, denominator=5)
    assert summary.attribution_coverage == _defined_ratio(numerator=3, denominator=4)
    assert [item.category for item in summary.category_counts] == [
        "BED_STALE_TRACK",
        "TRACKER_OR_IDENTITY",
        "UNCATEGORIZED",
    ]
    assert [item.count for item in summary.category_counts] == [1, 1, 1]
    assert all(item.ratio.denominator == 3 for item in summary.category_counts)
    assert all(item.ratio.numerator == item.count for item in summary.category_counts)
    assert summary.transport.unique_edge_event_count == 5
    assert summary.transport.total_attempts == 11
    assert summary.transport.extra_attempts_beyond_first == 6
    assert summary.transport.backend_event_id_available_count == 4
    assert summary.transport.distinct_backend_event_id_count == 3
    assert summary.transport.proof_backed_duplicate_count == 0
    assert summary.transport.proof_backed_retry_count == 0
    assert summary.transport.unique_alert_id.status == "UNAVAILABLE"
    assert summary.transport.unique_alert_id.value is None
    assert summary.transport.unique_alert_id.missing_reason == _ALERT_EXPORT_MISSING


def test_empty_cohort_keeps_zero_counts_and_null_ratio_boundaries() -> None:
    """An empty current-FP cohort cannot invent rates or alert IDs.

    Given no events and no exclusions
    When summarize_attribution_metrics runs
    Then counts are zero and every ratio is typed unavailable.
    """

    summary = _summarize(())

    _assert_partition(summary, attributable=0, pruned=0, unknown=0)
    assert summary.legacy_excluded_count == 0
    assert summary.legacy_excluded_census == {}
    assert summary.attribution_rate == _unavailable_ratio(
        numerator=0,
        reason="cohort_total_zero",
    )
    assert summary.retention_coverage == _unavailable_ratio(
        numerator=0,
        reason="cohort_total_zero",
    )
    assert summary.attribution_coverage == _unavailable_ratio(
        numerator=0,
        reason="evaluable_total_zero",
    )
    assert summary.category_counts == ()
    assert summary.transport.unique_edge_event_count == 0
    assert summary.transport.total_attempts == 0
    assert summary.transport.extra_attempts_beyond_first == 0
    assert summary.transport.backend_event_id_available_count == 0
    assert summary.transport.unique_alert_id.missing_reason == _ALERT_EXPORT_MISSING


def test_all_pruned_cohort_has_null_attribution_rate_among_evaluable() -> None:
    """All-pruned current FP is a retention loss, not a causal diagnosis.

    Given two PRUNED evidence records
    When metrics are summarized
    Then attributable and unknown stay zero and attribution coverage is unavailable.
    """

    summary = _summarize((_pruned_record(event_id="event:a"), _pruned_record(event_id="event:b")))

    _assert_partition(summary, attributable=0, pruned=2, unknown=0)
    assert summary.attribution_rate == _defined_ratio(numerator=0, denominator=2)
    assert summary.retention_coverage == _defined_ratio(numerator=0, denominator=2)
    assert summary.attribution_coverage == _unavailable_ratio(
        numerator=0,
        reason="evaluable_total_zero",
    )
    assert summary.category_counts == ()
    assert summary.transport.unique_edge_event_count == 2
    assert summary.transport.total_attempts == 4


def test_all_unknown_cohort_keeps_category_ratios_unavailable() -> None:
    """Complete retention without a category cannot produce category shares.

    Given two UNKNOWN evidence records
    When metrics are summarized
    Then attribution rate is zero over two and every category ratio is unavailable.
    """

    summary = _summarize((_unknown_record(event_id="event:a"), _unknown_record(event_id="event:b")))

    _assert_partition(summary, attributable=0, pruned=0, unknown=2)
    assert summary.attribution_rate == _defined_ratio(numerator=0, denominator=2)
    assert summary.retention_coverage == _defined_ratio(numerator=2, denominator=2)
    assert summary.attribution_coverage == _defined_ratio(numerator=0, denominator=2)
    assert summary.category_counts == ()


def test_all_attributable_cohort_uses_attributable_denominator_for_shares() -> None:
    """Every supported category share uses attributable_count as denominator.

    Given two COMPLETE classified events
    When metrics are summarized
    Then both category ratios use denominator 2.
    """

    summary = _summarize(
        (
            _classified(_stale_record(event_id="event:a")),
            _classified(_tracker_record(event_id="event:b")),
        )
    )

    _assert_partition(summary, attributable=2, pruned=0, unknown=0)
    assert summary.attribution_rate == _defined_ratio(numerator=2, denominator=2)
    assert [item.category for item in summary.category_counts] == [
        "BED_STALE_TRACK",
        "TRACKER_OR_IDENTITY",
    ]
    assert all(
        item.ratio == _defined_ratio(numerator=1, denominator=2)
        for item in summary.category_counts
    )


def test_zero_attributable_keeps_category_ratio_null_even_with_legacy_census() -> None:
    """Legacy exclusions never become a category-share denominator.

    Given one pruned event and one unmappable exclusion
    When metrics are summarized
    Then category ratios stay unavailable and exclusions stay outside cohort_total.
    """

    summary = _summarize(
        (_pruned_record(),),
        exclusions=(FalsePositiveCohortExclusion("UNMAPPABLE_LEGACY"),),
    )

    _assert_partition(summary, attributable=0, pruned=1, unknown=0)
    assert summary.legacy_excluded_count == 1
    assert summary.legacy_excluded_census == {"UNMAPPABLE_LEGACY": 1}
    assert summary.category_counts == ()
    assert summary.attribution_rate.denominator == 1
    assert summary.attribution_coverage.missing_reason == "evaluable_total_zero"


def test_duplicate_event_input_is_rejected() -> None:
    """Duplicate edge_event_id input is inconsistent, not silently collapsed.

    Given two rows for the same event
    When summarize_attribution_metrics runs
    Then it raises rather than repairing unique-event counts.
    """

    import pytest

    first = _classified(_stale_record(event_id="event:dup"))
    second = _classified(_tracker_record(event_id="event:dup"))

    with pytest.raises(ValueError, match="duplicate_edge_event_id"):
        _summarize((first, second))


def test_partition_inconsistency_is_rejected() -> None:
    """A classified PRUNED row cannot be repaired into an attributable count.

    Given a PRUNED evidence record that illegally carries a category
    When summarize_attribution_metrics runs
    Then it raises rather than choosing a partition.
    """

    import pytest

    illegal = AttributionMetricEvent(
        edge_event_id="event:pruned",
        evidence_status="PRUNED",
        category="BED_STALE_TRACK",
        neighborhood_pruned=True,
        attempt_count=2,
        backend_event_ids=("backend:pruned",),
        correlation_status="absent",
        correlation_kind=None,
    )

    with pytest.raises(ValueError, match="partition_inconsistent"):
        _summarize((illegal,))


def test_multiple_attempts_remain_one_unique_event() -> None:
    """One event with N attempts never changes unique detection counts.

    Given one COMPLETE event with attempt_count=5
    When metrics are summarized
    Then unique_edge_event_count stays 1 and extra attempts stay transport-only.
    """

    summary = _summarize((_classified(_complete_record(attempt_count=5)),))

    _assert_partition(summary, attributable=1, pruned=0, unknown=0)
    assert summary.category_counts[0].count == 1
    assert summary.category_counts[0].ratio.denominator == 1
    assert summary.transport.unique_edge_event_count == 1
    assert summary.transport.total_attempts == 5
    assert summary.transport.extra_attempts_beyond_first == 4
    assert summary.transport.proof_backed_duplicate_count == 0
    assert summary.transport.proof_backed_retry_count == 0


def test_missing_backend_ids_do_not_fabricate_availability() -> None:
    """Absent backend event IDs stay a transport availability count, not zero events.

    Given one attributable event with no backend IDs
    When metrics are summarized
    Then unique events stay 1 and backend availability stays 0.
    """

    summary = _summarize((_classified(_complete_record(backend_event_ids=())),))

    assert summary.transport.unique_edge_event_count == 1
    assert summary.transport.backend_event_id_available_count == 0
    assert summary.transport.distinct_backend_event_id_count == 0
    assert summary.attribution_rate.denominator == 1


def test_missing_alert_export_is_typed_unavailable_never_zero() -> None:
    """Absence of an alert-correlation export is not an empty available set.

    Given a mixed cohort and no export argument
    When metrics are summarized
    Then unique_alert_id is UNAVAILABLE(alert_correlation_export_not_supplied).
    """

    summary = _summarize(_mixed_events())

    assert summary.transport.unique_alert_id.status == "UNAVAILABLE"
    assert summary.transport.unique_alert_id.value is None
    assert summary.transport.unique_alert_id.missing_reason == _ALERT_EXPORT_MISSING
    assert summary.transport.unique_alert_id.value != 0


def test_explicit_empty_alert_export_is_available_zero() -> None:
    """An explicit empty valid export may report available zero alert IDs.

    Given one attributable event and an empty allowlisted alert export
    When metrics are summarized
    Then unique_alert_id is available with value 0.
    """

    summary = _summarize(
        (_classified(_stale_record()),),
        alert_correlation_export=(),
    )

    assert summary.transport.unique_alert_id.status == "AVAILABLE"
    assert summary.transport.unique_alert_id.value == 0
    assert summary.transport.unique_alert_id.missing_reason is None
    assert summary.transport.unique_edge_event_count == 1


def test_category_total_mismatch_is_rejected() -> None:
    """Category counts that cannot sum to attributable_count are inconsistent.

    Given one classified event whose category is outside the closed vocabulary
    When summarize_attribution_metrics runs
    Then it raises rather than emitting a repaired table.
    """

    import pytest

    illegal = AttributionMetricEvent(
        edge_event_id="event:complete",
        evidence_status="COMPLETE",
        category="IDLE_STATIC",
        neighborhood_pruned=False,
        attempt_count=1,
        backend_event_ids=("backend:one",),
        correlation_status="absent",
        correlation_kind=None,
    )

    with pytest.raises(ValueError, match="category_total_mismatch"):
        _summarize((illegal,))


def test_proof_backed_transport_does_not_change_unique_event_or_ratio() -> None:
    """Proven duplicate/retry counts stay orthogonal to detection ratios.

    Given one event with five attempts plus a typed retry export
    When metrics are summarized
    Then unique events and category ratios stay 1 while proof-backed retry is 1.
    """

    record = _complete_record(attempt_count=5)
    summary = _summarize((_classified(record, _retry_export(count=5)),))
    unproven = _summarize((_classified(record),))

    assert summary.transport.unique_edge_event_count == 1
    assert unproven.transport.unique_edge_event_count == 1
    assert summary.category_counts[0].count == 1
    assert summary.category_counts[0].ratio.denominator == 1
    assert summary.transport.total_attempts == 5
    assert summary.transport.extra_attempts_beyond_first == 4
    assert summary.transport.proof_backed_retry_count == 1
    assert summary.transport.proof_backed_duplicate_count == 0
    assert unproven.transport.proof_backed_retry_count == 0
    assert summary.category_counts[0].category == "DELIVERY_RETRY"
    assert unproven.category_counts[0].category == "UNCATEGORIZED"


def test_untrusted_alert_export_cannot_become_available_zero() -> None:
    """Notes-bearing or malformed alert exports stay typed unavailable.

    Given one attributable event and a notes-bearing export
    When metrics are summarized
    Then unique_alert_id stays UNAVAILABLE and unique events stay 1.
    """

    summary = _summarize(
        (_classified(_stale_record()),),
        alert_correlation_export=(
            {
                "edge_event_id": "event:stale",
                "alert_id": "alert:one",
                "notes": "operator said this looked duplicated in the UI",
            },
        ),
    )

    assert summary.transport.unique_alert_id.status == "UNAVAILABLE"
    assert summary.transport.unique_alert_id.value is None
    assert summary.transport.unique_alert_id.missing_reason == _ALERT_EXPORT_MISSING
    assert summary.transport.unique_edge_event_count == 1


def test_deterministic_order_and_serialization() -> None:
    """The same typed cohort serializes identically regardless of input order.

    Given a mixed cohort supplied in two permutations
    When summarize_attribution_metrics and machine_bytes run
    Then category order is vocabulary order and the machine bytes match.
    """

    from worker.fp_attribution import metrics_machine_bytes

    first = _mixed_events()
    second = tuple(reversed(first))
    left = _summarize(first)
    right = _summarize(second)

    assert [item.category for item in left.category_counts] == [
        "BED_STALE_TRACK",
        "TRACKER_OR_IDENTITY",
        "UNCATEGORIZED",
    ]
    assert [item.category for item in left.category_counts] == [
        item.category for item in right.category_counts
    ]
    assert set(_CATEGORY_VOCABULARY).issuperset(item.category for item in left.category_counts)
    assert metrics_machine_bytes(left) == metrics_machine_bytes(right)
    assert b"IDLE_STATIC" not in metrics_machine_bytes(left)
    assert b"notes" not in metrics_machine_bytes(left)


def test_supplied_alert_ids_count_distinct_values_without_changing_events() -> None:
    """A valid export reports distinct alert IDs and never extra events.

    Given two attributable events and three export rows that collapse to two IDs
    When metrics are summarized
    Then unique_alert_id is 2 and unique_edge_event_count stays 2.
    """

    events = (
        _classified(_stale_record(event_id="event:a")),
        _classified(_tracker_record(event_id="event:b")),
    )
    summary = _summarize(
        events,
        alert_correlation_export=(
            {"edge_event_id": "event:a", "alert_id": "alert:one"},
            {"edge_event_id": "event:b", "alert_id": "alert:two"},
            {"edge_event_id": "event:a", "alert_id": "alert:one"},
        ),
    )

    assert summary.transport.unique_edge_event_count == 2
    assert summary.transport.unique_alert_id.status == "AVAILABLE"
    assert summary.transport.unique_alert_id.value == 2
    assert summary.attribution_rate.denominator == 2


def test_duplicate_export_proof_stays_orthogonal_to_detection_ratio() -> None:
    """A proven UI duplicate remains one unique detection event.

    Given one COMPLETE event and a typed duplicate export
    When metrics are summarized
    Then category share stays 1/1 and proof-backed duplicate count is 1.
    """

    record = _stale_record()
    export = _duplicate_export(edge_event_id=record.edge_event_id)
    summary = _summarize((_classified(record, export),))

    assert summary.transport.unique_edge_event_count == 1
    assert summary.category_counts[0].category == "BACKEND_OR_UI_DUPLICATE"
    assert summary.category_counts[0].ratio == _defined_ratio(numerator=1, denominator=1)
    assert summary.transport.proof_backed_duplicate_count == 1
    assert summary.transport.proof_backed_retry_count == 0
