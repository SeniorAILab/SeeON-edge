"""RED/GREEN differential parity: native association vs the Python tracker oracle.

`worker.native.deepstream.association` is the executable specification the
custom `GstBaseTransform` association stage implements. `GreedyIouTracker`
(`worker/pipeline/perception/tracker.py`) remains the oracle for differential
shadow comparison (Task 4 guardrail) -- it is imported directly here, never
duplicated. The candidate module is imported directly at module scope: before
it exists, this whole file fails collection with a genuine `ModuleNotFoundError`
naming the missing package, which is the real RED for this seam, not a
manufactured `pytest.fail`.

The candidate strategy owns its OWN greedy-match iteration and tie-break
order; it must not import `worker.pipeline.perception.features.geometry`'s
`greedy_match` (the exact routine the oracle's tracker calls), because sharing
that routine would let an order/tie regression move both sides together and
leave the differential comparator green. `test_comparator_catches_*` proves
the comparator would fail on an independent tie/order drift.
"""

from __future__ import annotations

from typing import Literal

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor, NightWindow
from worker.domains.fall import FallEventLatch
from worker.native.deepstream.association import (
    ACTIVE_ASSOCIATION_STRATEGY_ID,
    ASSOCIATION_STRATEGY_REGISTRY,
    LEGACY_GREEDY_BBOX_IOU_V1,
    POSE_AWARE_BBOX_IOU_V1,
    AssociationFrameTrace,
    AssociationStrategy,
    AssociationStrategyDisabledError,
    PoseAwareStrategyDisabledError,
    build_active_association_strategy,
    build_association_strategy,
    compare_traces,
)
from worker.native.deepstream.association.legacy_greedy_iou import LegacyGreedyBboxIouStrategy
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types import BusinessEvent, DecisionInput

_OWN_GREEDY_MODULE_MARKER = "worker.pipeline.perception.features.geometry"


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _trace_from_oracle(
    oracle: GreedyIouTracker, frames: tuple[tuple[BoundingBox, ...] | None, ...]
) -> tuple[AssociationFrameTrace, ...]:
    traces: list[AssociationFrameTrace] = []
    for boxes in frames:
        if boxes is None:
            oracle.coast()
            traces.append(AssociationFrameTrace(track_ids=(), live_ids=oracle.live_ids))
            continue
        track_ids = oracle.observe(boxes)
        traces.append(AssociationFrameTrace(track_ids=track_ids, live_ids=oracle.live_ids))
    return tuple(traces)


def _trace_from_native(
    frames: tuple[tuple[BoundingBox, ...] | None, ...]
) -> tuple[AssociationFrameTrace, ...]:
    strategy = build_active_association_strategy()
    traces: list[AssociationFrameTrace] = []
    for boxes in frames:
        if boxes is None:
            strategy.coast()
            traces.append(AssociationFrameTrace(track_ids=(), live_ids=strategy.live_ids))
            continue
        outcome = strategy.observe(boxes)
        traces.append(
            AssociationFrameTrace(track_ids=outcome.track_ids, live_ids=outcome.live_ids)
        )
    return tuple(traces)


def _assert_parity(frames: tuple[tuple[BoundingBox, ...] | None, ...]) -> None:
    reference = _trace_from_oracle(GreedyIouTracker(), frames)
    candidate = _trace_from_native(frames)
    mismatches = compare_traces(reference, candidate)
    assert mismatches == (), f"parity mismatches: {mismatches}"


# --------------------------------------------------------------------------
# Independence: the candidate must own its own greedy iteration, not share
# the oracle's `greedy_match` routine.
# --------------------------------------------------------------------------


def test_legacy_strategy_module_does_not_import_the_oracles_greedy_match() -> None:
    import ast
    import inspect

    module = __import__(
        "worker.native.deepstream.association.legacy_greedy_iou",
        fromlist=["LegacyGreedyBboxIouStrategy"],
    )
    tree = ast.parse(inspect.getsource(module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    assert _OWN_GREEDY_MODULE_MARKER not in imported_modules, (
        "candidate strategy must not import the oracle tracker's own "
        "greedy_match/geometry module -- it must own independent greedy "
        "iteration so an order/tie regression cannot move both sides together"
    )
    source = inspect.getsource(LegacyGreedyBboxIouStrategy.observe)
    assert "for" in source or "sort" in source, "observe() must run its own matching loop"


# --------------------------------------------------------------------------
# Registry / cutover-gate: exactly one active identity.
# --------------------------------------------------------------------------


def test_only_legacy_greedy_strategy_is_active_at_cutover() -> None:
    assert ACTIVE_ASSOCIATION_STRATEGY_ID == LEGACY_GREEDY_BBOX_IOU_V1
    active = tuple(
        identity for identity, reg in ASSOCIATION_STRATEGY_REGISTRY.items() if reg.enabled
    )
    assert active == (LEGACY_GREEDY_BBOX_IOU_V1,)
    disabled = tuple(
        identity for identity, reg in ASSOCIATION_STRATEGY_REGISTRY.items() if not reg.enabled
    )
    assert POSE_AWARE_BBOX_IOU_V1 in disabled


def test_pose_aware_strategy_is_registered_but_refuses_to_activate() -> None:
    with pytest.raises(AssociationStrategyDisabledError):
        build_association_strategy(POSE_AWARE_BBOX_IOU_V1)


def test_pose_aware_strategy_instance_refuses_every_call() -> None:
    from worker.native.deepstream.association import PoseAwareAssociationStrategy

    strategy = PoseAwareAssociationStrategy()
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.observe((_box(0, 0, 10, 10),))
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.coast()
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.reset()


def test_legacy_strategy_satisfies_the_association_strategy_protocol() -> None:
    strategy = build_active_association_strategy()
    assert isinstance(strategy, AssociationStrategy)
    assert strategy.identity == LEGACY_GREEDY_BBOX_IOU_V1


# --------------------------------------------------------------------------
# Differential parity: greedy order, ties, skip/empty, eviction.
# --------------------------------------------------------------------------


def test_native_matches_oracle_on_greedy_order_and_box_order_return() -> None:
    first = _box(0, 0, 50, 50)
    second = _box(200, 200, 250, 250)
    moved_second = _box(205, 205, 255, 255)
    moved_first = _box(4, 4, 54, 54)
    _assert_parity(((first, second), (moved_second, moved_first)))


def test_native_matches_oracle_on_equal_iou_tie_keeping_lower_track_index() -> None:
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    center = _box(25, 0, 125, 100)
    _assert_parity(((left, right), (center,)))


def test_native_matches_oracle_on_inferred_empty_counts_a_miss() -> None:
    box = _box(0, 0, 20, 20)
    _assert_parity(((box,), (), (), (box,)))


def test_native_matches_oracle_on_skipped_frame_coasts_without_a_miss() -> None:
    box = _box(0, 0, 20, 20)
    _assert_parity(((box,), None, None, (box,)))


def test_native_matches_oracle_on_eviction_after_max_misses() -> None:
    box = _box(0, 0, 20, 20)
    oracle = GreedyIouTracker(max_misses=1)
    native = LegacyGreedyBboxIouStrategy(max_misses=1)

    assert oracle.observe((box,)) == (0,)
    assert native.observe((box,)).track_ids == (0,)
    assert oracle.observe(()) == ()
    assert native.observe(()).track_ids == ()
    assert oracle.observe(()) == ()
    assert native.observe(()).track_ids == ()
    assert oracle.live_ids == frozenset()
    assert native.live_ids == frozenset()
    assert oracle.observe((box,)) == (1,)
    assert native.observe((box,)).track_ids == (1,)


def test_native_matches_oracle_on_missing_pose_never_creates_or_evicts() -> None:
    """A frame with no pose/person cues at all (None) never mutates tracks."""
    person = _box(10, 10, 40, 40)
    _assert_parity(((person,), None, None, None, (person,)))


def test_native_matches_oracle_on_multi_person_crossing_and_occlusion() -> None:
    """Two people cross paths with real per-frame IoU overlap and briefly merge."""
    left = _box(0, 0, 40, 100)
    right = _box(80, 0, 120, 100)
    closing_left = _box(20, 0, 60, 100)
    closing_right = _box(60, 0, 100, 100)
    near_overlap_left = _box(30, 0, 70, 100)
    near_overlap_right = _box(50, 0, 90, 100)
    occluded_single = _box(35, 0, 85, 100)
    reemerging_left = _box(20, 0, 60, 100)
    reemerging_right = _box(60, 0, 100, 100)
    _assert_parity(
        (
            (left, right),
            (closing_left, closing_right),
            (near_overlap_left, near_overlap_right),
            (occluded_single,),
            None,
            (reemerging_left, reemerging_right),
        )
    )


def test_native_reset_matches_a_fresh_oracle_instance_after_reconnect() -> None:
    """Reconnect / stream-epoch rollover: reset must not resume prior ids."""
    box = _box(0, 0, 20, 20)
    native = build_active_association_strategy()
    _ = native.observe((box,))
    _ = native.observe((box,))
    assert native.live_ids == frozenset({0})

    native.reset()

    fresh_oracle = GreedyIouTracker()
    assert native.live_ids == frozenset()
    assert native.observe((box,)).track_ids == fresh_oracle.observe((box,))
    assert native.live_ids == fresh_oracle.live_ids == frozenset({0})


def test_boot_reset_cannot_observe_prior_tracker_state() -> None:
    """A fresh boot id must start from empty state, proven by two independent instances."""
    box = _box(0, 0, 20, 20)
    previous_boot = build_active_association_strategy()
    _ = previous_boot.observe((box,))
    _ = previous_boot.observe((box,))
    assert previous_boot.live_ids == frozenset({0})

    next_boot = build_active_association_strategy()
    assert next_boot.live_ids == frozenset()
    assert next_boot.observe((box,)).track_ids == (0,)


# --------------------------------------------------------------------------
# Malformed cue indexes and bed-overlap: association never treats bed as identity.
# --------------------------------------------------------------------------


def test_bed_only_boxes_never_resolve_to_the_prior_person_id_by_special_case() -> None:
    """The strategy treats every box cue uniformly; it does not know "bed"."""
    native = build_active_association_strategy()
    person = _box(10, 10, 40, 40)
    assert native.observe((person,)).track_ids == (0,)
    far_away_shape = _box(500, 500, 508, 508)
    outcome = native.observe((far_away_shape,))
    assert outcome.track_ids != (0,), "a non-overlapping cue must mint a new id, not reuse 0"


def test_selected_cue_indexes_must_address_person_box_cues_not_track_ids() -> None:
    """Malformed cue index: an index beyond the emitted box count is invalid."""
    from worker.types.perception_frame import (
        AssociationResult,
        PerceptionFrameIdentity,
        association_failure,
    )

    identity = PerceptionFrameIdentity(
        worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
    )
    malformed = AssociationResult(
        strategy=LEGACY_GREEDY_BBOX_IOU_V1,
        track_ids=(0,),
        selected_cue_indexes=(5,),
        identity=identity,
    )
    failure = association_failure(identity, malformed, person_box_count=1)
    assert failure is not None
    assert failure.code == "invalid_cue_index"


# --------------------------------------------------------------------------
# Downstream fall/bed-exit event timelines through the native strategy.
# --------------------------------------------------------------------------


class _FallModelMetadata:
    window = 1
    stride = 1
    mode: Literal["features", "sequence"] = "sequence"


class _FallModel:
    metadata = _FallModelMetadata()
    operating_threshold = 0.5

    def predict(self, features: object) -> float:
        del features
        return 0.9


def _fall_decision_input(track_ids: tuple[int, ...], frame_index: int) -> DecisionInput:
    pose = tuple((90, 50, 0.9) for _ in range(17))
    return DecisionInput(
        observation=FrameObservation(poses=(pose,), track_ids=track_ids),
        frame_width=200,
        frame_height=200,
        live_track_ids=track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_fall_event_timeline_is_identical_whether_oracle_or_native_supplies_track_ids() -> None:
    boxes = (_box(70, 20, 110, 130),)
    oracle = GreedyIouTracker()
    native = build_active_association_strategy()
    oracle_latch = FallEventLatch(_FallModel(), camera_id="camera-1", facility_id="facility-1")
    native_latch = FallEventLatch(_FallModel(), camera_id="camera-1", facility_id="facility-1")

    oracle_events: list[tuple[BusinessEvent, ...]] = []
    native_events: list[tuple[BusinessEvent, ...]] = []
    for frame_index in range(3):
        oracle_track_ids = oracle.observe(boxes)
        native_track_ids = native.observe(boxes).track_ids
        oracle_events.append(
            oracle_latch.update(_fall_decision_input(oracle_track_ids, frame_index))
        )
        native_events.append(
            native_latch.update(_fall_decision_input(native_track_ids, frame_index))
        )

    assert tuple(oracle_events) == tuple(native_events)
    assert any(events for events in oracle_events), "fixture must actually raise a fall event"


def _bed_exit_monitor() -> BedExitMonitor:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )


def _bed_exit_decision_input(
    track_ids: tuple[int, ...],
    boxes: tuple[BoundingBox, ...],
    bed: BoundingBox,
    frame_index: int,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=(boxes, ()),
            regions=((bed,), ()),
            track_ids=track_ids,
        ),
        frame_width=300,
        frame_height=300,
        live_track_ids=track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def test_bed_exit_event_timeline_is_identical_whether_oracle_or_native_supplies_track_ids() -> None:
    bed = _box(0, 0, 80, 100)
    in_bed = _box(10, 10, 70, 90)
    step = _box(10, 30, 70, 110)
    outside = _box(10, 70, 70, 150)
    frames = (in_bed, step, outside)

    oracle = GreedyIouTracker()
    native = build_active_association_strategy()
    oracle_monitor = _bed_exit_monitor()
    native_monitor = _bed_exit_monitor()

    oracle_events: list[tuple[BusinessEvent, ...]] = []
    native_events: list[tuple[BusinessEvent, ...]] = []
    for frame_index, box in enumerate(frames):
        oracle_track_ids = oracle.observe((box,))
        native_track_ids = native.observe((box,)).track_ids
        oracle_events.append(
            oracle_monitor.update(
                _bed_exit_decision_input(oracle_track_ids, (box,), bed, frame_index)
            )
        )
        native_events.append(
            native_monitor.update(
                _bed_exit_decision_input(native_track_ids, (box,), bed, frame_index)
            )
        )

    assert tuple(oracle_events) == tuple(native_events)
    assert any(events for events in oracle_events), "fixture must actually raise a bed-exit event"


# --------------------------------------------------------------------------
# Adversarial: intentional strategy drift must fail the comparator.
# --------------------------------------------------------------------------


def test_comparator_catches_intentional_track_id_drift() -> None:
    reference = (AssociationFrameTrace(track_ids=(0,), live_ids=frozenset({0})),)
    drifted = (AssociationFrameTrace(track_ids=(1,), live_ids=frozenset({1})),)

    mismatches = compare_traces(reference, drifted)

    assert len(mismatches) == 2
    fields = {mismatch.field for mismatch in mismatches}
    assert fields == {"track_ids", "live_ids"}


def test_comparator_catches_intentional_equal_iou_tie_break_drift() -> None:
    """Mutating the tie-break rule (favor the higher index) must fail parity."""
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    center = _box(25, 0, 125, 100)

    reference = _trace_from_oracle(GreedyIouTracker(), ((left, right), (center,)))

    class _DriftedStrategy:
        """Wraps the real strategy but flips the tie-break on a single match.

        Composition, not subclassing: `LegacyGreedyBboxIouStrategy` is
        `@final` (the shipped candidate must not grow subclass seams).
        """

        def __init__(self) -> None:
            self._inner = LegacyGreedyBboxIouStrategy()
            self.identity = self._inner.identity

        @property
        def live_ids(self) -> frozenset[int]:
            return self._inner.live_ids

        def observe(self, boxes: tuple[BoundingBox, ...]):
            outcome = self._inner.observe(boxes)
            if len(boxes) == 1 and len(outcome.track_ids) == 1:
                # Intentional mutation: report the wrong (higher) existing
                # track index on a tie, instead of the shipped lower-index win.
                from worker.native.deepstream.association import AssociationOutcome

                return AssociationOutcome(
                    track_ids=(outcome.track_ids[0] + 1,), live_ids=outcome.live_ids
                )
            return outcome

        def coast(self) -> None:
            self._inner.coast()

        def reset(self) -> None:
            self._inner.reset()

    drifted_strategy = _DriftedStrategy()
    candidate: list[AssociationFrameTrace] = []
    for boxes in ((left, right), (center,)):
        outcome = drifted_strategy.observe(boxes)
        candidate.append(
            AssociationFrameTrace(track_ids=outcome.track_ids, live_ids=drifted_strategy.live_ids)
        )

    mismatches = compare_traces(reference, tuple(candidate))
    assert mismatches != ()
    assert any(mismatch.field == "track_ids" for mismatch in mismatches)


def test_comparator_catches_missing_reconnect_reset() -> None:
    """Intentional drift: a strategy that forgets to reset on reconnect.

    Two stale pre-reconnect tracks exist; the second sits where the first
    post-reconnect box reappears. A properly reset strategy clears the track
    list, so `reset()` followed by one observation mints a fresh id 0. A
    strategy that forgets to reset instead greedy-matches the second stale
    track and reports its old id 1 -- the mismatch the comparator must catch.
    """
    stale_first = _box(0, 0, 20, 20)
    stale_second = _box(200, 200, 220, 220)
    reappearing = _box(202, 202, 222, 222)

    reference_strategy = build_active_association_strategy()
    _ = reference_strategy.observe((stale_first, stale_second))
    reference_strategy.reset()
    reference_outcome = reference_strategy.observe((reappearing,))
    reference = (
        AssociationFrameTrace(
            track_ids=reference_outcome.track_ids, live_ids=reference_strategy.live_ids
        ),
    )

    never_reset_strategy = build_active_association_strategy()
    _ = never_reset_strategy.observe((stale_first, stale_second))
    # Intentional drift: skip reset() entirely across the reconnect boundary.
    drifted_outcome = never_reset_strategy.observe((reappearing,))
    candidate = (
        AssociationFrameTrace(
            track_ids=drifted_outcome.track_ids, live_ids=never_reset_strategy.live_ids
        ),
    )

    mismatches = compare_traces(reference, candidate)
    assert mismatches != ()
    assert any(mismatch.field == "track_ids" for mismatch in mismatches)


def test_comparator_catches_malformed_cue_index_drift() -> None:
    """Intentional drift: track_ids/selected_cue_indexes length mismatch."""
    from worker.types.perception_frame import (
        AssociationResult,
        PerceptionFrameIdentity,
        association_failure,
    )

    identity = PerceptionFrameIdentity(
        worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
    )
    good = AssociationResult(
        strategy=LEGACY_GREEDY_BBOX_IOU_V1,
        track_ids=(0, 1),
        selected_cue_indexes=(0, 1),
        identity=identity,
    )
    assert association_failure(identity, good, person_box_count=2) is None

    drifted = AssociationResult(
        strategy=LEGACY_GREEDY_BBOX_IOU_V1,
        track_ids=(0, 1),
        selected_cue_indexes=(0,),
        identity=identity,
    )
    failure = association_failure(identity, drifted, person_box_count=2)
    assert failure is not None
    assert failure.code == "invalid_cue_index"
