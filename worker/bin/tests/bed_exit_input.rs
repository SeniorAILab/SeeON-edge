use std::collections::BTreeMap;

use seeon_ml_worker::policy::bed_exit_input::{BedInputError, BedInputProducer};
use seeon_ml_worker::policy::ingest::ObservedPose;
use seeon_ml_worker::relay::cameras::BedZoneRegion;
use seeon_worker::bed_exit::{AssignmentSnapshot, BedExitCapacities, BedRegionCacheState};

#[path = "support/bed_input_fixture.rs"]
mod fixture;
use fixture::{box_value, frame, make_camera};

fn capacities(tracks: usize, observations: usize, beds: usize, points: usize) -> BedExitCapacities {
    BedExitCapacities {
        tracks,
        observations,
        beds,
        pose_features: observations,
        polygon_points: points,
        episodes: tracks,
    }
}

fn region(id: &str, points: &[(i128, i128)]) -> BedZoneRegion {
    BedZoneRegion {
        id: id.into(),
        polygon: points.to_vec(),
        origin: "manual".into(),
    }
}

fn pose(track_id: u64, bbox: [f64; 4], point: [f64; 2]) -> ObservedPose {
    ObservedPose {
        track_id,
        bbox,
        confidence: 0.75,
        keypoints: vec![[point[0], point[1], 1.0]; 17],
    }
}

#[test]
fn all_persisted_regions_keep_order_polygon_bounds_and_confidence() {
    let polygons = [
        vec![(4, 2), (8, 3), (7, 9)],
        vec![(20, 0), (30, 0), (30, 10), (20, 10)],
    ];
    let configured_camera = make_camera(
        vec![
            region("first", &polygons[0]),
            region("second", &polygons[1]),
        ],
        (Some(64), Some(64)),
    );
    let mut producer =
        BedInputProducer::new(&configured_camera, capacities(4, 4, 2, 4), 2_000).unwrap();
    let input = producer
        .build(&frame(64, 64, vec![]), &[], &BTreeMap::new())
        .unwrap();
    assert_eq!(input.bed_boxes.len(), 2);
    assert_eq!(
        input.bed_boxes[0].polygon,
        Some(vec![(4, 2), (8, 3), (7, 9)])
    );
    assert_eq!(
        (
            input.bed_boxes[0].x1,
            input.bed_boxes[0].y1,
            input.bed_boxes[0].x2,
            input.bed_boxes[0].y2
        ),
        (4, 2, 8, 9)
    );
    assert_eq!(
        input.bed_boxes[1].polygon,
        Some(vec![(20, 0), (30, 0), (30, 10), (20, 10)])
    );
    assert_eq!(
        (
            input.bed_boxes[1].x1,
            input.bed_boxes[1].y1,
            input.bed_boxes[1].x2,
            input.bed_boxes[1].y2
        ),
        (20, 0, 30, 10)
    );
    assert!(input.bed_boxes.iter().all(|bed| bed.confidence == 1.0));
    assert_eq!(input.bed_region.source, BedRegionCacheState::Fresh);
    assert_eq!(input.bed_region.empty_cycles, 0);
}

#[test]
fn empty_regions_clear_beds_and_keep_matched_separate_from_live_ids() {
    let mut producer = BedInputProducer::new(
        &make_camera(vec![], (None, None)),
        capacities(4, 4, 0, 0),
        0,
    )
    .unwrap();
    let input = producer
        .build(
            &frame(32, 24, vec![7, 9]),
            &[pose(7, [1.0, 2.0, 5.0, 8.0], [3.0, 4.0])],
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(input.bed_boxes.is_empty());
    assert_eq!(input.bed_region.source, BedRegionCacheState::Empty);
    assert_eq!(input.bed_region.empty_cycles, 0);
    assert_eq!(input.track_ids, vec![Some(7)]);
    assert_eq!(input.live_track_ids, vec![7, 9]);
    assert_eq!(input.containments, vec![Vec::<f64>::new()]);
    assert_eq!(input.time_sec, Some(-1.25));
    assert_eq!(input.frame_index, 17);
}

#[test]
fn posture_scales_with_both_image_sizes_but_containment_keeps_original_pixels() {
    let polygon = [(0, 0), (10, 0), (10, 10), (0, 10)];
    let configured_camera = make_camera(vec![region("bed", &polygon)], (Some(20), Some(10)));
    let mut scaled =
        BedInputProducer::new(&configured_camera, capacities(4, 4, 1, 4), 500).unwrap();
    let observation = [pose(12, [18.0, 4.0, 20.0, 6.0], [18.0, 5.0])];
    let input = scaled
        .build(&frame(40, 20, vec![12, 99]), &observation, &BTreeMap::new())
        .unwrap();
    assert_eq!(input.containments[0], vec![0.0]);
    assert!(input.bed_pose_features[0].hip_depth > 0.0);
    assert_eq!(input.track_ids, vec![Some(12)]);
    assert_eq!(input.live_track_ids, vec![12, 99]);

    let unscaled_camera = make_camera(vec![region("bed", &polygon)], (Some(20), None));
    let mut unscaled =
        BedInputProducer::new(&unscaled_camera, capacities(4, 4, 1, 4), 500).unwrap();
    let unchanged = unscaled
        .build(&frame(40, 20, vec![12]), &observation, &BTreeMap::new())
        .unwrap();
    assert_eq!(unchanged.containments[0], vec![0.0]);
    assert!(unchanged.bed_pose_features[0].hip_depth < 0.0);
}

#[test]
fn positive_polygon_containment_and_hip_are_real_geometry() {
    let configured_camera = make_camera(
        vec![region("bed", &[(0, 0), (10, 0), (10, 10), (0, 10)])],
        (Some(10), Some(10)),
    );
    let mut producer =
        BedInputProducer::new(&configured_camera, capacities(2, 2, 1, 4), 200).unwrap();
    let input = producer
        .build(
            &frame(10, 10, vec![3]),
            &[pose(3, [2.0, 2.0, 4.0, 4.0], [3.0, 3.0])],
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(input.containments[0], vec![1.0]);
    assert_eq!(input.bed_pose_features[0].bed_id, Some(0));
    assert!(input.bed_pose_features[0].hip_depth > 0.0);
}

#[test]
fn prior_rows_bind_exact_assignment_box_and_preserve_observation_order() {
    let mut producer = BedInputProducer::new(
        &make_camera(vec![], (None, None)),
        capacities(3, 3, 0, 0),
        0,
    )
    .unwrap();
    let exact = box_value(0, 0, 10, 10);
    let assignments = BTreeMap::from([(
        77,
        AssignmentSnapshot {
            last_box: Some(exact.clone()),
            ..AssignmentSnapshot::default()
        },
    )]);
    let input = producer
        .build(
            &frame(32, 32, vec![8, 9]),
            &[
                pose(8, [2.0, 2.0, 6.0, 8.0], [4.0, 4.0]),
                pose(9, [20.0, 20.0, 22.0, 22.0], [21.0, 21.0]),
            ],
            &assignments,
        )
        .unwrap();
    assert_eq!(input.prior_box_containments.len(), 1);
    assert_eq!(input.prior_box_containments[0].previous_track_id, 77);
    assert_eq!(input.prior_box_containments[0].previous_box, exact);
    assert_eq!(input.prior_box_containments[0].ratios, vec![1.0, 0.0]);
}

#[test]
fn posture_geometry_rebuilds_after_frame_size_changes() {
    let configured_camera = make_camera(
        vec![region("bed", &[(0, 0), (10, 0), (10, 10), (0, 10)])],
        (Some(20), Some(10)),
    );
    let mut producer =
        BedInputProducer::new(&configured_camera, capacities(2, 2, 1, 4), 500).unwrap();
    let observation = [pose(4, [14.0, 4.0, 16.0, 6.0], [15.0, 5.0])];
    let first = producer
        .build(&frame(20, 10, vec![4]), &observation, &BTreeMap::new())
        .unwrap();
    assert!(first.bed_pose_features[0].hip_depth < 0.0);
    let resized = producer
        .build(&frame(40, 20, vec![4]), &observation, &BTreeMap::new())
        .unwrap();
    assert!(resized.bed_pose_features[0].hip_depth > 0.0);
    assert_eq!(resized.containments[0], vec![0.0]);
}

#[test]
fn capacities_and_unrepresentable_values_are_refused_with_typed_errors() {
    let polygon = [(0, 0), (5, 0), (5, 5), (0, 5)];
    let configured_camera = make_camera(vec![region("bed", &polygon)], (Some(5), Some(5)));
    assert!(matches!(
        BedInputProducer::new(&configured_camera, capacities(2, 2, 2, 3), 100),
        Err(BedInputError::Capacity {
            field: "polygon_points",
            ..
        })
    ));
    assert!(matches!(
        BedInputProducer::new(&configured_camera, capacities(2, 2, 0, 4), 100),
        Err(BedInputError::Capacity { field: "beds", .. })
    ));
    assert!(matches!(
        BedInputProducer::new(&configured_camera, capacities(2, 2, 1, 4), 0),
        Err(BedInputError::Containment(_))
    ));
    let mut producer =
        BedInputProducer::new(&configured_camera, capacities(2, 2, 1, 4), 100).unwrap();
    assert!(matches!(
        producer.build(
            &frame(5, 5, vec![1]),
            &[pose(1, [0.5, 0.0, 2.0, 2.0], [1.0, 1.0])],
            &BTreeMap::new(),
        ),
        Err(BedInputError::Conversion("person_bbox"))
    ));
    assert!(matches!(
        BedInputProducer::new(
            &make_camera(
                vec![region(
                    "wide",
                    &[(i128::from(i64::MAX) + 1, 0), (2, 0), (2, 1)]
                )],
                (None, None)
            ),
            capacities(2, 2, 1, 3),
            10,
        ),
        Err(BedInputError::Conversion("bed_coordinate"))
    ));
}
