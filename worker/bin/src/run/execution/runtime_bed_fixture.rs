use crate::policy::bed_exit::BedStage;
use crate::policy::bed_exit_input::BedInputProducer;
use crate::relay::cameras::{BedZoneRegion, RuntimeCamera};
use seeon_worker::bed_exit::{BedExitCapacities, BedExitConfig, BedExitMonitor};

pub(super) fn stage(episodes: usize, gated: bool) -> BedStage {
    let camera = RuntimeCamera {
        camera_id: "bed-camera".into(),
        facility_id: "facility".into(),
        rtsp_url: "rtsp://camera".into(),
        fps: 30.0,
        frame_stride: 1,
        decode_backend: None,
        label: None,
        bed_zone_regions: vec![BedZoneRegion {
            id: "bed".into(),
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
        episodes,
    };
    let inputs = BedInputProducer::new(&camera, limits, 1024).unwrap();
    let monitor = BedExitMonitor::new(
        BedExitConfig {
            camera_id: camera.camera_id.clone(),
            facility_id: camera.facility_id.clone(),
            min_containment: 0.35,
            hold_frames: 2,
            grace_frames: 3,
            night_window: gated.then(|| {
                seeon_worker::detection_window::DetectionWindow::from_zoneinfo_dir(
                    "12:00",
                    "13:00",
                    "UTC",
                    std::path::Path::new("/usr/share/zoneinfo"),
                )
                .unwrap()
            }),
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
    BedStage::new(monitor, inputs)
}
