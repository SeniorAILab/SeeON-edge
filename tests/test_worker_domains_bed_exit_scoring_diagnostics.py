from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from tests_support.bed_pose_fixtures import frame_pose_features, lying_in_bed
from worker.domains import bed_exit
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import DecisionInput
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures

CAMERA_ID: Final = "camera-bed-exit-scoring"
FACILITY_ID: Final = "facility-bed-exit-scoring"
PERSON_ID: Final = 3
BED_A: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED_A: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE_BEDS: Final = BoundingBox(40, 120, 100, 190, 0.94)


def _clock_at(hour: int = 22, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _monitor(
    *,
    hold_frames: int = 1,
    grace_frames: int = 2,
    in_bed_dwell_sec: float = 1.0,
    outside_dwell_sec: float = 1.0,
    scoring_recorder: bed_exit.BedExitScoringRecorder | None = None,
) -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            in_bed_dwell_sec=in_bed_dwell_sec,
            outside_dwell_sec=outside_dwell_sec,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=_clock_at(),
        scoring_recorder=scoring_recorder,
        boot_id="boot-bed-exit-scoring",
        stream_epoch="epoch-bed-exit-scoring",
        source_generation=0,
    )


def _input(
    *,
    person_boxes: tuple[BoundingBox, ...],
    bed_boxes: tuple[BoundingBox, ...],
    track_ids: tuple[int | None, ...],
    frame_index: int,
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=(person_boxes, ()),
            regions=(bed_boxes, ()),
            track_ids=track_ids,
        ),
        frame_width=200,
        frame_height=200,
        live_track_ids=tuple(track_id for track_id in track_ids if track_id is not None),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        bed_pose_features=bed_pose_features,
    )


def _lying_pose() -> FrameBedPoseFeatures:
    return frame_pose_features(lying_in_bed(track_id=PERSON_ID, bed_id=0))


def test_update_without_a_recorder_does_not_crash() -> None:
    monitor = _monitor()

    result = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )

    assert result == ()


def test_never_near_a_bed_reports_near_zero_containment_and_no_assignment() -> None:
    diagnostics = WorkerDiagnostics()
    monitor = _monitor(scoring_recorder=diagnostics)

    for frame_index in range(3):
        _ = monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )

    scoring = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert scoring is not None
    assert scoring.max_containment_observed == 0.0
    assert scoring.assignments_made == 0
    assert scoring.grace_positive_transitions == 0


def test_assignment_and_exit_are_both_reflected_cumulatively() -> None:
    diagnostics = WorkerDiagnostics()
    monitor = _monitor(grace_frames=2, scoring_recorder=diagnostics)

    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )
    after_assignment = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert after_assignment is not None
    assert after_assignment.max_containment_observed == 1.0
    assert after_assignment.assignments_made == 1
    assert after_assignment.grace_positive_transitions == 0

    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=1,
            bed_pose_features=_lying_pose(),
        )
    )
    after_arm = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert after_arm is not None
    assert after_arm.grace_positive_transitions == 1

    for frame_index in (2, 3):
        _ = monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )

    after_exit = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert after_exit is not None
    assert after_exit.max_containment_observed == 1.0
    assert after_exit.assignments_made == 1
    assert after_exit.grace_positive_transitions == 1


class _FailingScoringRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def record_bed_exit_scoring(
        self,
        camera_id: str,
        max_containment_observed: float,
        grace_positive_transitions: int,
        assignments_made: int,
    ) -> None:
        self.calls += 1
        raise RuntimeError("scoring telemetry sink is down")


def _exit_sequence() -> tuple[DecisionInput, ...]:
    return (
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        ),
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=1,
            bed_pose_features=_lying_pose(),
        ),
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=2,
        ),
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=3,
        ),
    )


def test_a_failing_scoring_recorder_never_blocks_the_bed_exit_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = _FailingScoringRecorder()
    failing = _monitor(grace_frames=2, scoring_recorder=recorder)
    baseline = _monitor(grace_frames=2)

    with caplog.at_level(logging.WARNING):
        emitted = [failing.update(frame) for frame in _exit_sequence()]
    expected = [baseline.update(frame) for frame in _exit_sequence()]

    assert recorder.calls == len(_exit_sequence())
    assert [
        [(event.event_type, event.person_id, event.bed_id, event.time_sec) for event in events]
        for events in emitted
    ] == [
        [(event.event_type, event.person_id, event.bed_id, event.time_sec) for event in events]
        for events in expected
    ]
    assert [event.event_type for events in emitted for event in events] == ["bed-exit"]
    contained = [
        record for record in caplog.records if "stage=bed_exit_scoring " in record.getMessage()
    ]
    assert len(contained) == 1
    assert f"camera_id={CAMERA_ID}" in contained[0].getMessage()
    assert not [record for record in caplog.records if record.exc_info]
