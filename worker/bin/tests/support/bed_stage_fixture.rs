use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::policy::bed_exit::{BedDecisionUpdate, BedStage, BedStageError};
use seeon_ml_worker::policy::bed_exit_input::BedInputProducer;
use seeon_ml_worker::policy::ingest::{Frame, ObservedPose};
use seeon_ml_worker::relay::cameras::BedZoneRegion;
use seeon_worker::bed_exit::{
    BedExitCapacities, BedExitClocks, BedExitConfig, BedExitMonitor, BedExitScoring,
};
use seeon_worker::detection_window::{AwareDateTime, DetectionWindow};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::trace::DecisionTraceSnapshot;

#[path = "bed_input_fixture.rs"]
mod input_fixture;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Receipt {
    pub frame: FrameIdentity,
    pub frame_index: i64,
    pub time_sec: Option<f64>,
    pub snapshots: Vec<DecisionTraceSnapshot>,
    pub events: Vec<BusinessEvent>,
    pub scoring: Option<BedExitScoring>,
    pub completed: bool,
}

impl From<BedDecisionUpdate<'_>> for Receipt {
    fn from(update: BedDecisionUpdate<'_>) -> Self {
        Self {
            frame: update.frame,
            frame_index: update.frame_index,
            time_sec: update.time_sec,
            snapshots: update.snapshots.to_vec(),
            events: update.events.to_vec(),
            scoring: update.scoring,
            completed: update.completed,
        }
    }
}

pub(super) fn capacities(episodes: usize) -> BedExitCapacities {
    BedExitCapacities {
        tracks: 4,
        observations: 4,
        beds: 1,
        pose_features: 4,
        polygon_points: 4,
        episodes,
    }
}

pub(super) fn room() -> BedZoneRegion {
    BedZoneRegion {
        id: "persisted-bed".into(),
        polygon: vec![(0, 0), (20, 0), (20, 20), (0, 20)],
        origin: "manual".into(),
    }
}

pub(super) fn stage(episodes: usize, regions: Vec<BedZoneRegion>) -> BedStage {
    configured_stage(episodes, regions, None)
}

pub(super) fn windowed_stage(episodes: usize, regions: Vec<BedZoneRegion>) -> BedStage {
    let window = DetectionWindow::from_zoneinfo_dir(
        "12:00",
        "13:00",
        "UTC",
        std::path::Path::new("/usr/share/zoneinfo"),
    )
    .unwrap();
    configured_stage(episodes, regions, Some(window))
}

pub(super) fn wall_at(hour: u64) -> AwareDateTime {
    AwareDateTime::from_utc_system_time(
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(hour * 3600),
    )
    .unwrap()
}

fn configured_stage(
    episodes: usize,
    regions: Vec<BedZoneRegion>,
    night_window: Option<DetectionWindow>,
) -> BedStage {
    let limits = capacities(episodes);
    let camera = input_fixture::make_camera(regions, (Some(32), Some(32)));
    let inputs = BedInputProducer::new(&camera, limits, 1_024).unwrap();
    let monitor = BedExitMonitor::new(
        BedExitConfig {
            camera_id: "camera-bed-stage".into(),
            facility_id: "facility-bed-stage".into(),
            min_containment: 0.5,
            hold_frames: 1,
            grace_frames: 0,
            night_window,
            in_bed_dwell_sec: 3.0,
            outside_dwell_sec: 2.0,
        },
        "boot-bed-stage",
        "stream-bed-stage",
        9,
        limits,
        3.0,
    )
    .unwrap();
    BedStage::new(monitor, inputs)
}

pub(super) fn pose(track_id: u64, bbox: [f64; 4]) -> ObservedPose {
    let point = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0];
    ObservedPose {
        track_id,
        bbox,
        confidence: 0.9,
        keypoints: vec![[point[0], point[1], 1.0]; 17],
    }
}

pub(super) fn frame_at(time: f64, index: i64, live: Vec<u64>) -> Frame {
    let mut frame = input_fixture::frame(32, 32, live);
    frame.time_sec = Some(time);
    frame.frame_index = index;
    frame.identity.sequence = 700 + index as u64;
    frame.identity.frame_number = index;
    frame
}

pub(super) fn submit(
    stage: &mut BedStage,
    time: f64,
    index: i64,
    live: Vec<u64>,
    poses: Vec<ObservedPose>,
) -> Result<Receipt, BedStageError> {
    let frame = frame_at(time, index, live);
    let mut receipt = None;
    stage.observe(
        &frame,
        &poses,
        BedExitClocks {
            wall_time: Some(wall_at(0)),
            observed_at: time,
            snapshot_at: time,
        },
        &mut |update| {
            assert!(receipt.is_none(), "observe must call its observer once");
            receipt = Some(Receipt::from(update));
        },
    )?;
    let receipt = receipt.expect("successful update has a receipt");
    assert_eq!(receipt.frame, frame.identity);
    assert_eq!(receipt.frame_index, frame.frame_index);
    assert_eq!(receipt.time_sec, frame.time_sec);
    assert_eq!(receipt.snapshots, stage.monitor().last_trace_snapshots());
    assert!(receipt.completed);
    Ok(receipt)
}

pub(super) fn inside(track: u64) -> ObservedPose {
    pose(track, [2.0, 2.0, 6.0, 8.0])
}

pub(super) fn outside(track: u64) -> ObservedPose {
    pose(track, [24.0, 2.0, 28.0, 8.0])
}

pub(super) fn arm_one(stage: &mut BedStage, tracks: &[u64]) {
    for (time, index) in [(0.0, 1), (1.0, 2), (2.0, 3), (3.0, 4)] {
        let poses = tracks.iter().copied().map(inside).collect();
        assert!(
            submit(stage, time, index, tracks.to_vec(), poses)
                .unwrap()
                .events
                .is_empty()
        );
    }
    assert!(
        tracks
            .iter()
            .all(|track| stage.monitor().assignments()[track].armed)
    );
}

pub(super) fn assert_retained_box(stage: &BedStage, track: u64, bounds: [i64; 4]) {
    let expected = input_fixture::box_value(bounds[0], bounds[1], bounds[2], bounds[3]);
    let actual = stage.monitor().assignments()[&track]
        .last_box
        .as_ref()
        .unwrap();
    assert_eq!(
        (actual.x1, actual.y1, actual.x2, actual.y2),
        (expected.x1, expected.y1, expected.x2, expected.y2)
    );
    assert_eq!(actual.polygon, expected.polygon);
}
