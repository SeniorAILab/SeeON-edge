from __future__ import annotations

from pathlib import Path

from test_fp_attribution_evidence import (
    _complete_seqs,
    _connect,
    _extract,
    _migrated,
    _record_for,
    _seed_fp_event,
)

from worker.fp_attribution import AttributionEvidenceRecord

NOW_BOOT = "boot:one"
NOW_EPOCH = 3


def _complete_record(**overrides: object) -> AttributionEvidenceRecord:
    payload: dict[str, object] = {
        "edge_event_id": "event:complete",
        "decision_reason": "fall-onset",
        "previous_state": "clear",
        "current_state": "fall",
        "score": 0.91,
        "threshold": 0.5,
        "score_missing_reason": None,
        "threshold_missing_reason": None,
        "track_id": 7,
        "track_missing_reason": None,
        "track_changed": False,
        "bed_id": None,
        "bed_missing_reason": "not-applicable",
        "bed_changed": False,
        "worker_boot_id": NOW_BOOT,
        "stream_epoch": NOW_EPOCH,
        "boot_changed": False,
        "epoch_changed": False,
        "associated_sibling_event_ids": (),
        "attempt_count": 1,
        "backend_event_ids": ("backend:one",),
        "coverage_status": "COMPLETE",
        "coverage_reason": None,
        "expected_frames": 30,
        "retained_frames": 30,
        "neighborhood_pruned": False,
        "evidence_status": "COMPLETE",
        "category": None,
        "prevented_eligible": True,
    }
    payload.update(overrides)
    return AttributionEvidenceRecord(**payload)


def _classify(record: AttributionEvidenceRecord, correlation: object | None = None):
    from worker.fp_attribution import classify_record

    return classify_record(record, correlation_export=correlation)


def _duplicate_export(
    *,
    edge_event_id: str = "event:complete",
    count: int = 2,
    kind: str = "BACKEND_OR_UI_DUPLICATE",
) -> dict[str, object]:
    return {
        "schema": "fp-correlation-v1",
        "edge_event_id": edge_event_id,
        "kind": kind,
        "user_visible_delivery_count": count,
    }


def _retry_export(*, edge_event_id: str = "event:complete", count: int = 3) -> dict[str, object]:
    return _duplicate_export(edge_event_id=edge_event_id, count=count, kind="DELIVERY_RETRY")


def _transport_export(*, edge_event_id: str = "event:complete") -> dict[str, object]:
    return {
        "schema": "fp-correlation-v1",
        "edge_event_id": edge_event_id,
        "kind": "TRANSPORT_ONLY",
        "user_visible_delivery_count": 1,
    }


def _decode_export(
    *,
    edge_event_id: str = "event:complete",
    fault: str | None = "decode-fault",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "fp-correlation-v1",
        "edge_event_id": edge_event_id,
        "kind": "CAMERA_LIGHTING_OR_DECODE",
    }
    if fault is not None:
        payload["typed_fault_code"] = fault
    return payload


def test_todo11_complete_pruned_and_unknown_statuses_keep_category_null(
    tmp_path: Path,
) -> None:
    """Characterize Todo 11 statuses before any classifier exists.

    Given current-FP rows with complete, pruned, and unknown evidence
    When attribution evidence is extracted
    Then statuses stay COMPLETE/PRUNED/UNKNOWN and category remains null.
    """

    database = _migrated(tmp_path)
    with _connect(database) as connection:
        complete_id = _seed_fp_event(
            connection,
            suffix="complete",
            seqs=_complete_seqs(),
        )
        pruned_id = _seed_fp_event(
            connection,
            suffix="pruned",
            seqs=tuple(range(71, 100)),
            trigger_seq=99,
        )
        unknown_id = _seed_fp_event(
            connection,
            suffix="unknown",
            seqs=tuple(range(200, 230)),
            trigger_seq=229,
            values=None,
        )
        connection.commit()

    result = _extract(database)
    complete = _record_for(result, complete_id)
    pruned = _record_for(result, pruned_id)
    unknown = _record_for(result, unknown_id)

    assert complete.evidence_status == "COMPLETE"
    assert complete.prevented_eligible is True
    assert complete.category is None
    assert pruned.evidence_status == "PRUNED"
    assert pruned.neighborhood_pruned is True
    assert pruned.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert pruned.category is None
    assert pruned.prevented_eligible is False
    assert unknown.evidence_status == "UNKNOWN"
    assert unknown.coverage_status == "COMPLETE"
    assert unknown.category is None
    assert unknown.prevented_eligible is False


def test_predicate_registry_is_an_explicit_ordered_table() -> None:
    """The owner-visible table is ordered and not a dict insertion policy.

    Given the public predicate registry
    When its declared categories are read
    Then first-match order is exactly the required vocabulary plus proof-gated delivery.
    """

    from worker.fp_attribution import PREDICATE_REGISTRY

    assert isinstance(PREDICATE_REGISTRY, tuple)
    assert tuple(spec.category for spec in PREDICATE_REGISTRY) == (
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
    assert not isinstance(PREDICATE_REGISTRY, dict)


def test_pruned_and_unknown_records_stay_unclassified() -> None:
    """Ineligible Todo 11 rows never receive a category.

    Given PRUNED and UNKNOWN evidence records
    When classify_record runs, even with a duplicate export
    Then category stays null and status/reason are preserved.
    """

    pruned = _complete_record(
        edge_event_id="event:pruned",
        evidence_status="PRUNED",
        coverage_status="GAP",
        coverage_reason="NEIGHBORHOOD_GAP_UNEXPLAINED",
        retained_frames=29,
        neighborhood_pruned=True,
        prevented_eligible=False,
        decision_reason="stale-track-exit",
        track_changed=True,
    )
    unknown = _complete_record(
        edge_event_id="event:unknown",
        evidence_status="UNKNOWN",
        score=None,
        threshold=None,
        score_missing_reason="value_not_persisted",
        threshold_missing_reason="value_not_persisted",
        prevented_eligible=False,
        decision_reason="stale-track-exit",
    )

    pruned_decision = _classify(pruned, _duplicate_export(edge_event_id="event:pruned"))
    unknown_decision = _classify(unknown)

    assert pruned_decision.category is None
    assert pruned_decision.evidence_status == "PRUNED"
    assert pruned_decision.annotations.coverage_reason == "NEIGHBORHOOD_GAP_UNEXPLAINED"
    assert pruned_decision.annotations.neighborhood_pruned is True
    assert unknown_decision.category is None
    assert unknown_decision.evidence_status == "UNKNOWN"
    assert unknown_decision.annotations.coverage_status == "COMPLETE"


def test_bed_stale_track_wins_tracker_collision() -> None:
    """Stale-track reason beats a simultaneous track-identity change.

    Given COMPLETE stale-track-exit evidence with track_changed
    When classify_record runs
    Then the category is BED_STALE_TRACK.
    """

    record = _complete_record(
        decision_reason="stale-track-exit",
        previous_state="live-grace",
        current_state="triggered",
        track_changed=True,
        bed_id=2,
        bed_missing_reason=None,
    )

    decision = _classify(record)

    assert decision.category == "BED_STALE_TRACK"
    assert decision.evidence_status == "COMPLETE"
    assert decision.annotations.matched_predicate == "BED_STALE_TRACK"


def test_fall_latch_rearm_wins_episode_collision() -> None:
    """Same-interval fall-onset siblings are latch rearm, not fragmentation.

    Given COMPLETE fall-onset evidence with an explicit sibling
    When classify_record runs
    Then the category is FALL_LATCH_REARM.
    """

    record = _complete_record(
        decision_reason="fall-onset",
        previous_state="clear",
        current_state="fall",
        associated_sibling_event_ids=("event:sibling",),
        track_changed=True,
    )

    decision = _classify(record)

    assert decision.category == "FALL_LATCH_REARM"
    assert decision.annotations.matched_predicate == "FALL_LATCH_REARM"


def test_episode_fragmentation_wins_tracker_collision() -> None:
    """Explicit siblings that are not a fall rearm stay fragmentation.

    Given COMPLETE non-onset evidence with siblings and track_changed
    When classify_record runs
    Then the category is EPISODE_FRAGMENTATION.
    """

    record = _complete_record(
        decision_reason="fall-active",
        previous_state="fall",
        current_state="fall",
        associated_sibling_event_ids=("event:sibling-a", "event:sibling-b"),
        track_changed=True,
    )

    decision = _classify(record)

    assert decision.category == "EPISODE_FRAGMENTATION"
    assert decision.annotations.matched_predicate == "EPISODE_FRAGMENTATION"


def test_zero_or_missing_pose_wins_geometry_collision() -> None:
    """Typed pose-absence beats a simultaneous bed-assignment change.

    Given COMPLETE person-observation-missing evidence with bed_changed
    When classify_record runs
    Then the category is ZERO_OR_MISSING_POSE.
    """

    record = _complete_record(
        decision_reason="person-observation-missing",
        previous_state="unknown",
        current_state="no-decision",
        track_id=None,
        track_missing_reason="no-observed-person",
        bed_changed=True,
        bed_id=1,
        bed_missing_reason=None,
    )

    decision = _classify(record)

    assert decision.category == "ZERO_OR_MISSING_POSE"
    assert decision.annotations.matched_predicate == "ZERO_OR_MISSING_POSE"


def test_camera_lighting_or_decode_requires_typed_fault_fact() -> None:
    """Decode/lighting is coded only from an allowlisted typed fault.

    Given COMPLETE otherwise-uncategorized evidence
    When classify_record runs with and without a typed decode fault
    Then only the coded fault becomes CAMERA_LIGHTING_OR_DECODE.
    """

    record = _complete_record()

    coded = _classify(record, _decode_export())
    absent = _classify(record)
    prose = _classify(
        record,
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:complete",
            "kind": "CAMERA_LIGHTING_OR_DECODE",
            "notes": "the room looked dark and the decoder skipped frames",
        },
    )

    assert coded.category == "CAMERA_LIGHTING_OR_DECODE"
    assert absent.category == "UNCATEGORIZED"
    assert prose.category == "UNCATEGORIZED"
    assert prose.annotations.correlation_status == "rejected"


def test_insufficient_evidence_does_not_fall_through_to_uncategorized() -> None:
    """A higher applicable but unevaluable predicate stops classification.

    Given COMPLETE fall-onset evidence whose previous_state is missing
    When classify_record runs
    Then the category is INSUFFICIENT_EVIDENCE, not UNCATEGORIZED.
    """

    insufficient = _complete_record(previous_state=None)
    uncategorized = _complete_record()

    stopped = _classify(insufficient)
    fallback = _classify(uncategorized)

    assert stopped.category == "INSUFFICIENT_EVIDENCE"
    assert stopped.annotations.matched_predicate == "INSUFFICIENT_EVIDENCE"
    assert fallback.category == "UNCATEGORIZED"
    assert fallback.annotations.matched_predicate == "UNCATEGORIZED"


def test_transport_only_requires_typed_transport_proof() -> None:
    """Transport-only is a first-match category only with typed proof.

    Given COMPLETE uncategorized detection facts plus a transport-only export
    When classify_record runs
    Then the category is TRANSPORT_ONLY.
    """

    record = _complete_record(attempt_count=4)

    decision = _classify(record, _transport_export())

    assert decision.category == "TRANSPORT_ONLY"
    assert decision.annotations.attempt_count == 4
    assert decision.annotations.matched_predicate == "TRANSPORT_ONLY"


def test_backend_or_ui_duplicate_requires_repeated_user_visible_proof() -> None:
    """A typed duplicate export can select BACKEND_OR_UI_DUPLICATE.

    Given COMPLETE stale-track evidence and a valid duplicate export
    When classify_record runs
    Then the proven delivery category wins over detection.
    """

    record = _complete_record(
        decision_reason="stale-track-exit",
        previous_state="live-grace",
        current_state="triggered",
    )

    decision = _classify(record, _duplicate_export(count=2))

    assert decision.category == "BACKEND_OR_UI_DUPLICATE"
    assert decision.annotations.correlation_status == "accepted"
    assert decision.annotations.correlation_kind == "BACKEND_OR_UI_DUPLICATE"
    assert decision.annotations.matched_predicate == "BACKEND_OR_UI_DUPLICATE"


def test_delivery_retry_requires_repeated_user_visible_proof() -> None:
    """A typed retry export can select DELIVERY_RETRY.

    Given COMPLETE uncategorized evidence and a valid retry export
    When classify_record runs
    Then the category is DELIVERY_RETRY.
    """

    record = _complete_record(attempt_count=3)

    decision = _classify(record, _retry_export(count=3))

    assert decision.category == "DELIVERY_RETRY"
    assert decision.annotations.correlation_status == "accepted"
    assert decision.annotations.correlation_kind == "DELIVERY_RETRY"
    assert decision.annotations.attempt_count == 3


def test_high_attempt_count_without_correlation_stays_detection_or_uncategorized() -> None:
    """attempt_count and backend IDs never choose a delivery category.

    Given COMPLETE rows with high attempt_count and no correlation export
    When classify_record runs
    Then stale-track stays a detection category and bare facts stay UNCATEGORIZED.
    """

    stale = _complete_record(
        decision_reason="stale-track-clear",
        previous_state="contained",
        current_state="retired",
        attempt_count=9,
        backend_event_ids=("backend:one", "backend:two"),
    )
    bare = _complete_record(attempt_count=9, backend_event_ids=("backend:one",))

    stale_decision = _classify(stale)
    bare_decision = _classify(bare)

    assert stale_decision.category == "BED_STALE_TRACK"
    assert bare_decision.category == "UNCATEGORIZED"
    assert stale_decision.annotations.attempt_count == 9
    assert bare_decision.annotations.attempt_count == 9
    assert stale_decision.annotations.backend_event_ids == ("backend:one", "backend:two")
    assert stale_decision.annotations.correlation_status == "absent"
    assert bare_decision.annotations.correlation_status == "absent"


def test_false_and_malformed_correlation_exports_are_rejected() -> None:
    """False or malformed correlation proof cannot select a delivery category.

    Given COMPLETE uncategorized evidence and illegal exports
    When classify_record runs
    Then each payload is rejected and the category stays UNCATEGORIZED.
    """

    record = _complete_record(attempt_count=6)
    rejected_payloads = (
        True,
        "BACKEND_OR_UI_DUPLICATE",
        {"duplicate": True, "attempt_count": 6},
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:complete",
            "kind": "BACKEND_OR_UI_DUPLICATE",
            "user_visible_delivery_count": 1,
        },
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:other",
            "kind": "DELIVERY_RETRY",
            "user_visible_delivery_count": 4,
        },
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:complete",
            "kind": "DELIVERY_RETRY",
            "user_visible_delivery_count": "3",
        },
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:complete",
            "kind": "NOT_A_KIND",
            "user_visible_delivery_count": 4,
        },
        {
            "schema": "fp-correlation-v1",
            "edge_event_id": "event:complete",
            "kind": "BACKEND_OR_UI_DUPLICATE",
            "user_visible_delivery_count": 4,
            "notes": "operator said this looked duplicated in the UI",
        },
    )

    for payload in rejected_payloads:
        decision = _classify(record, payload)
        assert decision.category == "UNCATEGORIZED"
        assert decision.annotations.correlation_status == "rejected"
        assert decision.annotations.correlation_kind is None
        assert decision.annotations.correlation_rejection_reason is not None


def test_same_input_repeated_is_byte_identical() -> None:
    """Classification is a pure function of allowlisted facts.

    Given the same COMPLETE record and retry export twice
    When classify_record and machine_bytes run
    Then the machine bytes are identical.
    """

    from worker.fp_attribution import machine_bytes

    record = _complete_record(
        associated_sibling_event_ids=("event:b", "event:a"),
        attempt_count=2,
        backend_event_ids=("backend:z", "backend:a"),
    )
    export = _retry_export(count=2)

    first = _classify(record, export)
    second = _classify(record, export)

    assert machine_bytes(first) == machine_bytes(second)
    assert machine_bytes(first) == machine_bytes(_classify(record, dict(export)))


def test_tracker_pose_geometry_and_uncategorized_positive_matches() -> None:
    """Each remaining detection category has a single positive fixture.

    Given isolated COMPLETE facts for tracker, geometry, and bare detection
    When classify_record runs
    Then each fixture returns exactly one category.
    """

    tracker = _classify(_complete_record(track_changed=True, decision_reason="fall-active"))
    geometry = _classify(
        _complete_record(
            decision_reason="contained-in-other-bed",
            previous_state="contained",
            current_state="other-bed",
            bed_id=2,
            bed_missing_reason=None,
            bed_changed=True,
        )
    )
    bare = _classify(_complete_record())

    assert tracker.category == "TRACKER_OR_IDENTITY"
    assert geometry.category == "BED_GEOMETRY_OR_ASSIGNMENT"
    assert bare.category == "UNCATEGORIZED"


def test_unsupported_plan_categories_are_never_emitted() -> None:
    """Idle/model/annotation tokens stay outside the public vocabulary.

    Given the public registry and a bare COMPLETE record
    When classify_record runs
    Then unsupported plan categories are absent from the result and table.
    """

    from worker.fp_attribution import PREDICATE_REGISTRY

    decision = _classify(_complete_record())
    rendered = repr(decision) + "".join(spec.category for spec in PREDICATE_REGISTRY)

    assert decision.category == "UNCATEGORIZED"
    for token in ("IDLE_STATIC", "MODEL_OR_THRESHOLD", "ANNOTATION_ERROR"):
        assert token not in rendered


def test_extracted_domain_evidence_does_not_change_terminal_categories(
    tmp_path: Path,
) -> None:
    """R5 domain facts stay orthogonal to the current classifier vocabulary.

    Given an extracted COMPLETE fall record that now carries typed latch/gap facts
    When classify_record runs without a correlation export
    Then the terminal category remains UNCATEGORIZED.
    """

    from test_fp_attribution_evidence import _pose_components

    database = _migrated(tmp_path)
    seqs = _complete_seqs()
    with _connect(database) as connection:
        edge_event_id = _seed_fp_event(
            connection,
            suffix="classify-domain",
            seqs=seqs,
            analysis_track_by_seq={seq: None if seq >= 36 else 7 for seq in seqs},
            components_by_seq={
                seq: _pose_components("not-scheduled" if seq in {20, 21} else "observed")
                for seq in seqs
            },
            neighborhood_decisions=(
                {
                    "seq": 30,
                    "reason": "fall-onset",
                    "previous_state": "clear",
                    "current_state": "fall",
                    "triggered": 1,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.88, None),),
                },
                {
                    "seq": 33,
                    "reason": "below-threshold",
                    "previous_state": "fall",
                    "current_state": "clear",
                    "triggered": 0,
                    "track_id": 7,
                    "module_qualified_id": "fall.v1",
                    "policy_qualified_id": "fall.policy.v1",
                    "values": (("fall_probability", 0.12, None),),
                },
            ),
        )
        connection.commit()

    record = _record_for(_extract(database), edge_event_id)
    decision = _classify(record)

    assert record.person_presence.status == "PERSON_GAP"
    assert record.fall_latch.status == "AVAILABLE"
    assert decision.category == "UNCATEGORIZED"
    assert decision.evidence_status == "COMPLETE"
