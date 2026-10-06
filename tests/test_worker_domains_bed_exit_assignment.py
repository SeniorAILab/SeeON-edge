from __future__ import annotations

from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from tests_support.bed_pose_fixtures import frame_pose_features, lying_in_bed
from worker.domains import bed_exit
from worker.types import DecisionInput
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures

PERSON_ID: Final = 7
BED: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE_BED: Final = BoundingBox(100, 10, 160, 90, 0.94)


def _monitor(
    *,
    camera_id: str,
    hold_frames: int = 1,
    grace_frames: int = 0,
    in_bed_dwell_sec: float = 1.0,
    outside_dwell_sec: float = 1.0,
) -> bed_exit.BedExitMonitor:
    fixed = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=camera_id,
            facility_id="facility-bed-exit",
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            in_bed_dwell_sec=in_bed_dwell_sec,
            outside_dwell_sec=outside_dwell_sec,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: fixed,
        boot_id=f"boot-{camera_id}",
        stream_epoch=f"epoch-{camera_id}",
        source_generation=0,
    )


def _input(
    person: BoundingBox,
    beds: tuple[BoundingBox, ...],
    frame_index: int,
    *,
    live_track_ids: tuple[int, ...] = (PERSON_ID,),
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=(beds, ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=180,
        frame_height=120,
        live_track_ids=live_track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        bed_pose_features=bed_pose_features,
    )


def _lying_pose() -> FrameBedPoseFeatures:
    return frame_pose_features(lying_in_bed(track_id=PERSON_ID, bed_id=0))


def test_hold_and_containment_tie_assign_the_lowest_bed_id() -> None:
    monitor = _monitor(camera_id="camera-tie", hold_frames=2)
    overlapping_beds = (BED, BED)

    first = monitor.update(_input(IN_BED, overlapping_beds, 0))
    second = monitor.update(_input(IN_BED, overlapping_beds, 1))

    assert first == second == ()
    assert monitor.last_debug_snapshot is not None
    assert tuple(status.occupancy for status in monitor.last_debug_snapshot.statuses) == (
        "occupied",
        "empty",
    )


def test_detector_instances_keep_camera_assignments_isolated() -> None:
    camera_a = _monitor(camera_id="camera-a")
    camera_b = _monitor(camera_id="camera-b")
    _ = camera_a.update(_input(IN_BED, (BED,), 0))
    _ = camera_a.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))

    camera_b_events = camera_b.update(_input(OUTSIDE_BED, (BED,), 2))
    camera_a_events = camera_a.update(_input(OUTSIDE_BED, (BED,), 2))

    assert camera_b_events == ()
    assert len(camera_a_events) == 1
    assert camera_a_events[0].camera_id == "camera-a"


def test_track_lost_mid_grace_never_emits_on_absence() -> None:
    monitor = _monitor(camera_id="camera-lost-mid-exit", hold_frames=1, grace_frames=2)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    events = monitor.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    assert events == ()


def test_single_sub_threshold_frame_before_track_loss_no_longer_fires() -> None:
    monitor = _monitor(camera_id="camera-single-frame-jitter", hold_frames=1, grace_frames=3)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    events = monitor.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    assert events == ()


def test_dead_observed_track_cannot_emit_after_identity_reuse() -> None:
    monitor = _monitor(camera_id="camera-reused-track")
    _ = monitor.update(_input(IN_BED, (BED,), 0))

    dead_track = monitor.update(_input(IN_BED, (BED,), 1, live_track_ids=()))
    reused_identity = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    assert dead_track == reused_identity == ()


def test_a_released_bed_exit_does_not_rearm_on_track_loss() -> None:
    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager

    monitor = _monitor(camera_id="camera-lost-mid-exit", hold_frames=1, grace_frames=2)
    aggregator = EventAggregator(deciders=(monitor,), incidents=IncidentManager(cooldown_sec=300.0))
    aggregator.update(_input(IN_BED, (BED,), 0))
    aggregator.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))

    exited = aggregator.update(_input(OUTSIDE_BED, (BED,), 2))
    assert len(exited) == 1, "the armed exit was never reported"

    aggregator.release(exited[0])

    repeated = aggregator.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    assert repeated == ()


def test_track_id_churn_during_continuous_occupancy_preserves_arm_progress() -> None:
    SUCCESSOR_ID: Final = 8

    monitor = _monitor(camera_id="camera-churn", hold_frames=1)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor._assignments[PERSON_ID].armed is True

    churned = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((IN_BED,), ()),
                regions=((BED,), ()),
                track_ids=(SUCCESSOR_ID,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(SUCCESSOR_ID,),
            time_sec=2.0,
            frame_index=2,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
            bed_pose_features=frame_pose_features(
                lying_in_bed(track_id=SUCCESSOR_ID, bed_id=0)
            ),
        )
    )

    assert churned == ()
    assert PERSON_ID not in monitor._assignments
    assert monitor._assignments[SUCCESSOR_ID].armed is True

    exited = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((OUTSIDE_BED,), ()),
                regions=((BED,), ()),
                track_ids=(SUCCESSOR_ID,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(SUCCESSOR_ID,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert len(exited) == 1
    assert exited[0].person_id == SUCCESSOR_ID
    assert exited[0].bed_id == 0


def test_track_id_churn_during_outside_dwell_carries_progress_and_fires_once() -> None:
    monitor = _monitor(camera_id="camera-unrelated-churn", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()

    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()
    successor_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((OUTSIDE_BED,), ()),
                regions=((BED,), ()),
                track_ids=(successor_id,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(successor_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert len(events) == 1
    assert events[0].person_id == successor_id
    assert events[0].bed_id == 0
    assert PERSON_ID not in monitor._assignments
    assert monitor._assignments[successor_id].armed is False
    assert monitor._assignments[successor_id].outside_dwell_sec == 0.0


def test_new_track_contained_in_a_different_bed_does_not_inherit_outside_dwell() -> None:
    bed_two: Final = BoundingBox(200, 10, 260, 90, 0.94)
    monitor = _monitor(camera_id="camera-two-bed-churn", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED, bed_two), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED, bed_two), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor.update(_input(OUTSIDE_BED, (BED, bed_two), 2)) == ()

    other_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((bed_two,), ()),
                regions=((BED, bed_two), ()),
                track_ids=(other_id,),
            ),
            frame_width=300,
            frame_height=120,
            live_track_ids=(other_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert events == ()
    assert PERSON_ID not in monitor._assignments
    assert monitor._assignments[other_id].armed is False
    assert monitor._assignments[other_id].outside_dwell_sec == 0.0


def test_unrelated_new_track_far_away_does_not_inherit_outside_dwell() -> None:
    door: Final = BoundingBox(300, 10, 360, 90, 0.9)
    monitor = _monitor(camera_id="camera-caregiver-at-door", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()

    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()
    caregiver_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((door,), ()),
                regions=((BED,), ()),
                track_ids=(caregiver_id,),
            ),
            frame_width=400,
            frame_height=120,
            live_track_ids=(caregiver_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert events == ()
    assert PERSON_ID not in monitor._assignments
    assert monitor._assignments[caregiver_id].armed is False
    assert monitor._assignments[caregiver_id].outside_dwell_sec == 0.0


def test_outside_handoff_blocked_when_the_vacated_bed_is_reoccupied() -> None:
    door: Final = BoundingBox(300, 10, 360, 90, 0.9)
    monitor = _monitor(
        camera_id="camera-reentry-blocks-handoff", hold_frames=1, outside_dwell_sec=2.0
    )
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()

    reentry_id = 10
    door_id = 11
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((IN_BED, door), ()),
                regions=((BED,), ()),
                track_ids=(reentry_id, door_id),
            ),
            frame_width=400,
            frame_height=120,
            live_track_ids=(reentry_id, door_id),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert events == ()
    assert PERSON_ID not in monitor._assignments
    assert monitor._assignments[reentry_id].armed is False
    assert monitor._assignments[door_id].armed is False
    assert monitor._assignments[door_id].outside_dwell_sec == 0.0
