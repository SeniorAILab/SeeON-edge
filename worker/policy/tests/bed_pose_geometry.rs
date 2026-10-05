use seeon_worker::bed_exit::{BedPoseFeatures, BoundingBox};
use seeon_worker::bed_pose_geometry::{BedPoseGeometry, PoseGeometryError};

fn box_region(x1: i64, y1: i64, x2: i64, y2: i64) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        confidence: 1.0,
        polygon: None,
    }
}

fn explicit(points: &[(i64, i64)]) -> BoundingBox {
    let mut region = box_region(0, 0, 10, 10);
    region.polygon = Some(points.to_vec());
    region
}

fn geometry(beds: &[BoundingBox], polygon_size: Option<[i64; 2]>) -> BedPoseGeometry {
    BedPoseGeometry::new(beds, [100, 100], polygon_size, 16, 4096).unwrap()
}

fn pose(x: f64, y: f64) -> Vec<[f64; 3]> {
    vec![[x, y, 1.0]; 17]
}

fn features(value: &BedPoseGeometry, pose: &[[f64; 3]]) -> BedPoseFeatures {
    value.features(42, pose).unwrap()
}

#[test]
fn signed_hip_depth_uses_inside_and_outside_distance() {
    let geometry = geometry(&[box_region(0, 0, 10, 10)], None);
    let expected = 5.0 / 200.0_f64.sqrt();
    let inside = features(&geometry, &pose(5.0, 5.0));
    let outside = features(&geometry, &pose(15.0, 5.0));
    let boundary = features(&geometry, &pose(0.0, 5.0));
    let right_boundary = features(&geometry, &pose(10.0, 5.0));
    assert!((inside.hip_depth - expected).abs() < 1e-12);
    assert!((outside.hip_depth + expected).abs() < 1e-12);
    assert_eq!(inside.keypoint_in_frac, 1.0);
    assert_eq!(outside.keypoint_in_frac, 0.0);
    assert_eq!(inside.hip_x_rel, 0.5);
    assert_eq!(inside.hip_y_rel, 0.5);
    assert_eq!(inside.centroid_displacement, 0.0);
    assert_eq!(boundary.keypoint_in_frac, 1.0);
    assert_eq!(boundary.hip_depth, 0.0);
    assert_eq!(right_boundary.keypoint_in_frac, 0.0);
    assert_eq!(right_boundary.hip_depth, 0.0);
    assert!(right_boundary.hip_depth.is_sign_negative());
}

#[test]
fn torso_angle_is_the_absolute_hip_to_shoulder_atan2() {
    let geometry = geometry(&[box_region(-10, -10, 10, 10)], None);
    let mut person = vec![[0.0, 0.0, 0.0]; 17];
    person[5] = [3.0, 4.0, 1.0];
    person[11] = [0.0, 0.0, 1.0];
    let metrics = features(&geometry, &person);
    assert!((metrics.torso_angle - 4.0_f64.atan2(3.0)).abs() < 1e-12);
}

#[test]
fn polygon_axes_scale_independently_and_original_index_wins_ties() {
    let bed = explicit(&[(0, 0), (100, 0), (100, 100), (0, 100)]);
    let scaled =
        BedPoseGeometry::new(&[bed.clone()], [200, 100], Some([100, 100]), 16, 4096).unwrap();
    let mut point = pose(150.0, 50.0);
    let metrics = features(&scaled, &point);
    assert_eq!(metrics.bed_id, Some(0));
    assert_eq!(metrics.hip_x_rel, 0.75);
    assert_eq!(metrics.hip_y_rel, 0.5);
    assert!(metrics.hip_depth > 0.0);

    let tied = geometry(&[bed.clone(), bed], None);
    assert_eq!(features(&tied, &pose(50.0, 50.0)).bed_id, Some(0));
    point.push([f64::NAN, 0.0, 1.0]);
    assert_eq!(features(&scaled, &point).bed_id, Some(0));
}

#[test]
fn torso_priority_precedes_broader_keypoint_coverage() {
    let beds = [box_region(0, 0, 10, 10), box_region(20, 0, 30, 10)];
    let geometry = geometry(&beds, None);
    let mut person = pose(5.0, 5.0);
    for index in [5, 6, 11, 12] {
        person[index] = [25.0, 5.0, 1.0];
    }
    let selected = features(&geometry, &person);
    assert_eq!(selected.bed_id, Some(1));
    assert_eq!(selected.torso_in_frac, 1.0);
    assert!(selected.keypoint_in_frac < 1.0);
}

#[test]
fn explicit_invalid_polygon_never_uses_bbox_but_missing_polygon_does() {
    let malformed = explicit(&[(0, 0), (10, 0)]);
    let missing = box_region(0, 0, 10, 10);
    let invalid = features(&geometry(&[malformed], None), &pose(5.0, 5.0));
    let fallback = features(&geometry(&[missing], None), &pose(5.0, 5.0));
    assert_eq!(invalid.bed_id, None);
    assert!(!invalid.bed_polygon_valid);
    assert_eq!(fallback.bed_id, Some(0));

    let collinear = explicit(&[(0, 0), (5, 5), (10, 10)]);
    assert_eq!(
        features(&geometry(&[collinear], None), &pose(5.0, 5.0)).bed_id,
        None
    );
    let repeated = explicit(&[(0, 0), (10, 0), (10, 0), (10, 10), (0, 10), (0, 0)]);
    assert!(features(&geometry(&[repeated], None), &pose(5.0, 5.0)).bed_polygon_valid);
}

#[test]
fn confidence_boundary_missing_joints_and_missing_bed_are_explicit() {
    let with_bed = geometry(&[box_region(0, 0, 10, 10)], None);
    let mut person = vec![[0.0, 0.0, 0.0]; 17];
    person[11] = [5.0, 5.0, 0.5];
    person[12] = [5.0, 5.0, 0.499];
    let metrics = features(&with_bed, &person);
    assert_eq!(metrics.observability, 1.0 / 17.0);
    assert_eq!(metrics.keypoint_in_frac, 1.0);
    assert_eq!(metrics.torso_angle, 0.0);
    assert_eq!(metrics.hip_x_rel, 0.5);

    let empty = geometry(&[], None);
    let missing = features(&empty, &person);
    assert_eq!(missing.bed_id, None);
    assert!(!missing.bed_polygon_valid);
    assert_eq!(missing.observability, 1.0 / 17.0);
}

#[test]
fn budgets_nonpositive_scale_and_nonfinite_inputs_refuse_safely() {
    let bed = box_region(0, 0, 10, 10);
    assert_eq!(
        BedPoseGeometry::new(std::slice::from_ref(&bed), [100, 100], None, 0, 4).err(),
        Some(PoseGeometryError::Capacity)
    );
    assert_eq!(
        BedPoseGeometry::new(std::slice::from_ref(&bed), [100, 100], None, 1, 3).err(),
        Some(PoseGeometryError::Capacity)
    );
    let disabled = BedPoseGeometry::new(&[bed], [100, 100], Some([0, 100]), 1, 4).unwrap();
    assert_eq!(features(&disabled, &pose(5.0, 5.0)).bed_id, None);
    let disabled_frame = BedPoseGeometry::new(
        &[box_region(0, 0, 10, 10)],
        [100, 0],
        Some([100, 100]),
        1,
        4,
    )
    .unwrap();
    assert_eq!(features(&disabled_frame, &pose(5.0, 5.0)).bed_id, None);

    let valid = geometry(&[box_region(0, 0, 10, 10)], None);
    let mut invalid = pose(5.0, 5.0);
    invalid[0][2] = f64::NAN;
    assert_eq!(
        valid.features(42, &invalid).err(),
        Some(PoseGeometryError::NonFinite)
    );
    let huge = pose(f64::MAX, f64::MAX);
    assert_eq!(
        valid.features(42, &huge).err(),
        Some(PoseGeometryError::NonFinite)
    );
}
