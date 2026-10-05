use std::collections::BTreeMap;

use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::policy::ingest::Frame;
use seeon_ml_worker::relay::cameras::{BedZoneRegion, RuntimeCamera};
use seeon_worker::bed_exit::BoundingBox;

pub fn box_value(x1: i64, y1: i64, x2: i64, y2: i64) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        confidence: 0.4,
        polygon: None,
    }
}

pub fn make_camera(regions: Vec<BedZoneRegion>, size: (Option<u64>, Option<u64>)) -> RuntimeCamera {
    RuntimeCamera {
        camera_id: "camera".into(),
        facility_id: "facility".into(),
        rtsp_url: "rtsp://camera".into(),
        fps: 30.0,
        frame_stride: 1,
        decode_backend: None,
        label: None,
        bed_zone_regions: regions,
        bed_zone_image_width: size.0,
        bed_zone_image_height: size.1,
    }
}

pub fn frame(width: i64, height: i64, live_track_ids: Vec<u64>) -> Frame {
    Frame {
        identity: FrameIdentity::default(),
        width,
        height,
        live_track_ids,
        rows: BTreeMap::new(),
        time_sec: Some(-1.25),
        frame_index: 17,
    }
}
