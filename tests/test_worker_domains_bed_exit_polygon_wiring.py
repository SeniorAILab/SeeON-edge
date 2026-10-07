from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
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
from worker.pipeline.perception.scene_state import SceneState
from worker.types import DecisionInput
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures

PERSON_ID: Final = 1

DIAMOND_BED: Final = BoundingBox(
    x1=0,
    y1=0,
    x2=100,
    y2=100,
    confidence=1.0,
    polygon=((50, 0), (100, 50), (50, 100), (0, 50)),
)

IN_BED: Final = BoundingBox(40, 40, 60, 60, 0.95)

BESIDE_BED_IN_AABB: Final = BoundingBox(80, 80, 95, 95, 0.9)


def _monitor(
    *,
    min_containment: float = 0.5,
    grace_frames: int = 1,
    in_bed_dwell_sec: float = 1.0,
    outside_dwell_sec: float = 1.0,
) -> bed_exit.BedExitMonitor:
    fixed = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id="camera-polygon",
            facility_id="facility-polygon",
            min_containment=min_containment,
            hold_frames=1,
            grace_frames=grace_frames,
            in_bed_dwell_sec=in_bed_dwell_sec,
            outside_dwell_sec=outside_dwell_sec,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: fixed,
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )


def _input(
    person: BoundingBox,
    frame_index: int,
    *,
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((DIAMOND_BED,), ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=100,
        frame_height=100,
        live_track_ids=(PERSON_ID,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        bed_pose_features=bed_pose_features,
    )


def _lying_pose() -> FrameBedPoseFeatures:
    return frame_pose_features(lying_in_bed(track_id=PERSON_ID, bed_id=0))


def test_person_beside_polygon_bed_but_inside_aabb_eventually_exits() -> None:
    monitor = _monitor(grace_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, 0)) == ()
    assert monitor.update(_input(IN_BED, 1, bed_pose_features=_lying_pose())) == ()
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].occupancy == "occupied"

    frame_2 = monitor.update(_input(BESIDE_BED_IN_AABB, 2))
    frame_3 = monitor.update(_input(BESIDE_BED_IN_AABB, 3))

    assert frame_2 == ()
    assert len(frame_3) == 1
    assert frame_3[0].bed_id == 0
    assert frame_3[0].person_id == PERSON_ID


def test_person_beside_polygon_bed_reads_as_not_occupied_immediately() -> None:
    monitor = _monitor(grace_frames=1)
    assert monitor.update(_input(IN_BED, 0)) == ()

    _ = monitor.update(_input(BESIDE_BED_IN_AABB, 1))

    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].occupancy != "occupied"


def test_bed_exit_input_can_only_receive_persisted_polygons() -> None:
    scene = SceneState("camera-polygon")
    observation, _ = scene.resolve_bed_regions(
        FrameObservation(regions=((DIAMOND_BED,), ())),
        frame_index=0,
        bed_scheduled=True,
        bed_interval=1,
    )
    assert observation.bed_boxes == ()
    assert scene.bed_polygon_source == "none"

    tree = ast.parse(Path("worker/pipeline/perception/scene_state.py").read_text())
    segmentation_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "bed_boxes"
        and isinstance(node.value, ast.Name)
        and node.value.id == "observation"
    ]
    assert segmentation_reads == []
