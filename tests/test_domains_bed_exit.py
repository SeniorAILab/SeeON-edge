from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from edge.domains.bed_exit.detector import BedExitMonitor, NightWindow
from edge.domains.bed_exit.schema import BedExitEvent, BedExitFrame, BedStatus
from edge.perception.domain_input import DomainInput


def box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def update(
    monitor: BedExitMonitor,
    beds: tuple[BoundingBox, ...],
    persons: tuple[BoundingBox, ...],
) -> BedExitFrame:
    return monitor.update_boxes(bed_boxes=beds, person_boxes=persons)


def observation_for(person: BoundingBox, bed: BoundingBox) -> FrameObservation:
    return FrameObservation(detections=((person,), ()), regions=((bed,), ()))


def domain_input(observation: FrameObservation, time_sec: float) -> DomainInput:
    return DomainInput(
        observation=observation,
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=time_sec,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH, 0),
    )


def fixed_clock(hour: int, minute: int, second: int = 0):
    seoul = ZoneInfo("Asia/Seoul")
    return lambda: datetime(2026, 1, 1, hour, minute, second, tzinfo=seoul)


def runtime_exit_events(
    monitor: BedExitMonitor,
    *,
    bed: BoundingBox,
    time_sec: float = 1.0,
) -> tuple[dict[str, object], ...]:
    monitor.update(domain_input(observation_for(box(20, 0, 120, 100), bed), 0.0))
    return monitor.update(domain_input(observation_for(box(60, 0, 160, 100), bed), time_sec))


def test_schema_exports_bed_exit_frame_statuses_and_events() -> None:
    bed = box(0, 0, 100, 100)
    status = BedStatus(bed_id=0, box=bed, occupancy="exit", person_id=7)
    event = BedExitEvent(person_id=7, bed_id=0)
    assert BedExitFrame(statuses=(status,), events=(event,)).statuses == (status,)


def test_hold_frames_prevent_jitter_assignment_until_stable() -> None:
    beds = (box(0, 0, 100, 100), box(120, 0, 220, 100))
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=2, grace_frames=1)

    first = update(monitor, beds, (box(10, 10, 60, 60),))
    assert [s.occupancy for s in first.statuses] == ["empty", "empty"]

    jitter = update(monitor, beds, (box(130, 10, 180, 60),))
    assert [s.occupancy for s in jitter.statuses] == ["empty", "empty"]

    held_once = update(monitor, beds, (box(130, 10, 180, 60),))
    assert [s.occupancy for s in held_once.statuses] == ["empty", "occupied"]


def test_own_bed_exit_after_grace_emits_once_and_unassigns() -> None:
    beds = (box(0, 0, 100, 100),)
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=1, grace_frames=1)

    assigned = update(monitor, beds, (box(50, 10, 110, 50),))
    assert assigned.statuses[0].occupancy == "occupied"

    grace = update(monitor, beds, (box(75, 10, 135, 50),))
    assert grace.statuses[0].occupancy == "empty"
    assert grace.events == ()

    exited = update(monitor, beds, (box(90, 10, 150, 50),))
    assert [(e.person_id, e.bed_id) for e in exited.events] == [(0, 0)]
    assert exited.statuses[0].occupancy == "exit"

    after = update(monitor, beds, (box(90, 10, 150, 50),))
    assert after.events == ()


def test_cross_bed_movement_does_not_false_positive_own_bed_exit() -> None:
    beds = (box(0, 0, 100, 100), box(80, 0, 180, 100))
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=1, grace_frames=0)

    update(monitor, beds, (box(50, 10, 110, 70),))
    moved_to_other_bed = update(monitor, beds, (box(75, 10, 135, 70),))

    assert moved_to_other_bed.events == ()
    assert [s.occupancy for s in moved_to_other_bed.statuses] == ["empty", "empty"]


def test_overlapping_beds_choose_best_containment_with_lowest_id_tiebreak() -> None:
    beds = (box(0, 0, 100, 100), box(20, 0, 120, 100))
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=1)

    tied = update(monitor, beds, (box(20, 10, 100, 90),))

    assert [(s.bed_id, s.occupancy) for s in tied.statuses] == [
        (0, "occupied"),
        (1, "empty"),
    ]


def test_invalid_parameters_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="min_containment"):
        BedExitMonitor(min_containment=0.0)
    with pytest.raises(ValueError, match="hold_frames"):
        BedExitMonitor(hold_frames=0)
    with pytest.raises(ValueError, match="grace_frames"):
        BedExitMonitor(grace_frames=-1)


def test_runtime_observation_update_returns_domain_event_tuple() -> None:
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=1, grace_frames=0)
    bed = box(0, 0, 100, 100)

    assert (
        monitor.update(
            domain_input(
                FrameObservation(detections=((box(20, 0, 120, 100),), ()), regions=((bed,), ())),
                0.0,
            )
        )
        == ()
    )

    assert monitor.update(
        domain_input(
            FrameObservation(detections=((box(60, 0, 160, 100),), ()), regions=((bed,), ())),
            1.0,
        )
    ) == (
        {
            "domain": "bed_exit",
            "event_type": "bed-exit",
            "identity": "0:0",
            "person_id": 0,
            "bed_id": 0,
            "probability": 1.0,
            "time_sec": 1.0,
        },
    )


@pytest.mark.parametrize(
    ("hour", "minute", "second"),
    [
        (21, 0, 0),
        (4, 59, 59),
    ],
)
def test_night_window_cross_midnight_emits_inside_window(
    hour: int,
    minute: int,
    second: int,
) -> None:
    bed = box(0, 0, 100, 100)
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        clock=fixed_clock(hour, minute, second),
    )

    events = runtime_exit_events(monitor, bed=bed)

    assert len(events) == 1
    assert events[0]["event_type"] == "bed-exit"


@pytest.mark.parametrize(
    ("hour", "minute", "second"),
    [
        (20, 59, 59),
        (5, 0, 0),
    ],
)
def test_night_window_cross_midnight_suppresses_outside_window(
    hour: int,
    minute: int,
    second: int,
) -> None:
    bed = box(0, 0, 100, 100)
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        clock=fixed_clock(hour, minute, second),
    )

    assert runtime_exit_events(monitor, bed=bed) == ()


def test_night_window_uses_injected_clock_not_monotonic_time_sec() -> None:
    bed = box(0, 0, 100, 100)
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        clock=fixed_clock(13, 0, 0),
    )

    assert runtime_exit_events(monitor, bed=bed, time_sec=23 * 3600) == ()


def test_night_window_rejects_naive_clock_datetime() -> None:
    bed = box(0, 0, 100, 100)
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        clock=lambda: datetime(2026, 1, 1, 22, 0, 0),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        runtime_exit_events(monitor, bed=bed)


def test_without_night_window_emits_regardless_of_clock_time() -> None:
    bed = box(0, 0, 100, 100)
    monitor = BedExitMonitor(min_containment=0.5, hold_frames=1, grace_frames=0)

    events = runtime_exit_events(monitor, bed=bed)

    assert len(events) == 1
