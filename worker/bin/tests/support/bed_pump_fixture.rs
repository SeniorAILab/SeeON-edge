use std::time::{Duration, SystemTime};

use seeon_deepstream_native::{FrameIdentity, MediaBinding, TrackedObject};
use seeon_ml_worker::msg::PosePacket;
use seeon_ml_worker::policy::bed_exit::BedStage;
use seeon_ml_worker::policy::bed_exit_input::BedInputProducer;
use seeon_ml_worker::policy::fall::FallStage;
use seeon_ml_worker::relay::cameras::{BedZoneRegion, RuntimeCamera};
use seeon_ml_worker::run::pump::CameraPolicy;
use seeon_ml_worker::seam::Clock;
use seeon_worker::bed_exit::{BedExitCapacities, BedExitConfig, BedExitMonitor};
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};

pub(super) struct NoWallClock;
impl Clock for NoWallClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }
    fn wall(&self) -> SystemTime {
        panic!("ungated bed policy must not read wall time")
    }
    fn pause(&self, _: Duration) {}
}

pub(super) fn camera(fall: bool) -> CameraPolicy {
    let camera = RuntimeCamera {
        camera_id: "bed-camera".into(),
        facility_id: "facility".into(),
        rtsp_url: "rtsp://camera".into(),
        fps: 30.0,
        frame_stride: 1,
        decode_backend: None,
        label: None,
        bed_zone_regions: vec![BedZoneRegion {
            id: "persisted".into(),
            origin: "manual".into(),
            polygon: vec![(0, 0), (20, 0), (20, 20), (0, 20)],
        }],
        bed_zone_image_width: Some(32),
        bed_zone_image_height: Some(32),
    };
    let limits = BedExitCapacities {
        tracks: 4,
        observations: 4,
        beds: 1,
        pose_features: 4,
        polygon_points: 4,
        episodes: 8,
    };
    let inputs = BedInputProducer::new(&camera, limits, 1024).unwrap();
    let monitor = BedExitMonitor::new(
        BedExitConfig {
            camera_id: camera.camera_id.clone(),
            facility_id: camera.facility_id.clone(),
            min_containment: 0.35,
            hold_frames: 2,
            grace_frames: 3,
            night_window: None,
            in_bed_dwell_sec: 3.0,
            outside_dwell_sec: 2.0,
        },
        "boot",
        "1",
        1,
        limits,
        3.0,
    )
    .unwrap();
    let stage = fall.then(|| {
        let decider = FallPolicyDecider::new(
            "bed-camera",
            "facility",
            "boot",
            "1",
            1,
            FallPolicy::default(),
            FallCapacities {
                retained_tracks: 4,
                generation_identities: 4,
                episodes: 8,
                vote_window: 5,
            },
        )
        .unwrap();
        FallStage::new(decider, 1.0).unwrap()
    });
    CameraPolicy {
        source_id: 3,
        stage,
        bed_stage: Some(BedStage::new(monitor, inputs)),
    }
}

pub(super) fn packet(sequence: u64, outside: bool, missing: bool) -> PosePacket {
    let left = if outside { 24.0 } else { 2.0 };
    let mut row = [0.0; 57];
    row[..5].copy_from_slice(&[left, 2.0, left + 4.0, 8.0, 0.9]);
    // Tensor geometry uses the 640-square network; SDK boxes use the 32-square mux.
    for coordinate in &mut row[..4] {
        *coordinate *= 20.0;
    }
    for point in 0..17 {
        row[6 + 3 * point] = (left + 2.0) * 20.0;
        row[7 + 3 * point] = 5.0 * 20.0;
        row[8 + 3 * point] = 1.0;
    }
    PosePacket {
        frame: FrameIdentity {
            binding: MediaBinding {
                token: 4,
                generation: 1,
                epoch: 1,
            },
            source_id: 3,
            sequence,
            frame_number: sequence as i64,
            pts_ns: sequence * 1_000_000_000,
            pts_valid: 1,
            source_width: 32,
            source_height: 32,
            analysis_width: 32,
            analysis_height: 32,
            ..FrameIdentity::default()
        },
        tensor_present: true,
        rows: if missing { vec![] } else { vec![row] },
        objects: vec![TrackedObject {
            track_id: 7,
            left,
            top: 2.0,
            width: 4.0,
            height: 6.0,
            confidence: 0.9,
        }],
    }
}
