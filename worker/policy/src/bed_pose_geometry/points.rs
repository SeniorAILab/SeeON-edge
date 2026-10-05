use crate::bed_exit::BedPoseFeatures;

use super::PoseGeometryError;
use super::polygon::{CachedPolygon, contains, edge_distance, finite};

const KEYPOINTS: usize = 17;
const TORSO: [usize; 4] = [5, 6, 11, 12];
const LOWER: [usize; 4] = [13, 14, 15, 16];

pub(super) struct Person {
    valid: [bool; KEYPOINTS],
    points: [[f64; 2]; KEYPOINTS],
    count: usize,
    observability: f64,
    torso_angle: f64,
}

impl Person {
    pub(super) fn new(pose: &[[f64; 3]]) -> Result<Self, PoseGeometryError> {
        let mut person = Self {
            valid: [false; KEYPOINTS],
            points: [[0.0; 2]; KEYPOINTS],
            count: 0,
            observability: 0.0,
            torso_angle: 0.0,
        };
        for (index, keypoint) in pose.iter().take(KEYPOINTS).enumerate() {
            for value in keypoint {
                finite(*value)?;
            }
            if keypoint[2] >= 0.5 {
                person.valid[index] = true;
                person.points[index] = [keypoint[0], keypoint[1]];
                person.count += 1;
            }
        }
        person.observability = person.count as f64 / KEYPOINTS as f64;
        person.torso_angle = person.torso_angle()?;
        finite(person.observability)?;
        finite(person.torso_angle)?;
        Ok(person)
    }

    pub(super) fn against(
        &self,
        track_id: u64,
        polygon: &CachedPolygon,
    ) -> Result<BedPoseFeatures, PoseGeometryError> {
        let mut inside = [false; KEYPOINTS];
        for (index, value) in inside.iter_mut().enumerate() {
            if self.valid[index] {
                *value = contains(self.points[index], polygon)?;
            }
        }
        let torso_in_frac = fraction(&self.valid, &inside, &TORSO);
        let lower_in_frac = fraction(&self.valid, &inside, &LOWER);
        let mut keypoint_in_frac = 0.0;
        if self.count > 0 {
            let inside_count = (0..KEYPOINTS).filter(|index| inside[*index]).count();
            keypoint_in_frac = inside_count as f64 / self.count as f64;
        }

        let hip = self.midpoint(&[11, 12]);
        let (hip_depth, hip_x_rel, hip_y_rel) = match hip {
            Some(point) => {
                let distance = edge_distance(point, polygon)?;
                let sign = if contains(point, polygon)? { 1.0 } else { -1.0 };
                let depth = sign * distance / polygon.diagonal;
                let x_rel = (point[0] - polygon.min[0]) / polygon.span[0];
                let y_rel = (point[1] - polygon.min[1]) / polygon.span[1];
                (depth, x_rel, y_rel)
            }
            None => (0.0, 0.0, 0.0),
        };
        let centroid_displacement = self.centroid_displacement(polygon)?;
        let values = [
            torso_in_frac,
            lower_in_frac,
            keypoint_in_frac,
            hip_depth,
            self.torso_angle,
            centroid_displacement,
            hip_x_rel,
            hip_y_rel,
            self.observability,
        ];
        for value in values {
            finite(value)?;
        }
        Ok(BedPoseFeatures {
            track_id,
            bed_id: Some(polygon.bed_id),
            torso_in_frac,
            lower_in_frac,
            keypoint_in_frac,
            hip_depth,
            torso_angle: self.torso_angle,
            centroid_displacement,
            hip_x_rel,
            hip_y_rel,
            observability: self.observability,
            bed_polygon_valid: true,
        })
    }

    pub(super) fn missing_bed(&self, track_id: u64) -> BedPoseFeatures {
        BedPoseFeatures {
            track_id,
            bed_id: None,
            torso_in_frac: 0.0,
            lower_in_frac: 0.0,
            keypoint_in_frac: 0.0,
            hip_depth: 0.0,
            torso_angle: self.torso_angle,
            centroid_displacement: 0.0,
            hip_x_rel: 0.0,
            hip_y_rel: 0.0,
            observability: self.observability,
            bed_polygon_valid: false,
        }
    }

    fn torso_angle(&self) -> Result<f64, PoseGeometryError> {
        let Some(shoulder) = self.midpoint(&[5, 6]) else {
            return Ok(0.0);
        };
        let Some(hip) = self.midpoint(&[11, 12]) else {
            return Ok(0.0);
        };
        let dx = (shoulder[0] - hip[0]).abs();
        let dy = (shoulder[1] - hip[1]).abs();
        finite(dx)?;
        finite(dy)?;
        if dx == 0.0 && dy == 0.0 {
            Ok(0.0)
        } else {
            Ok(dy.atan2(dx))
        }
    }

    fn midpoint(&self, indices: &[usize]) -> Option<[f64; 2]> {
        let mut sum = [0.0, 0.0];
        let mut count = 0usize;
        for &index in indices {
            if self.valid[index] {
                sum[0] += self.points[index][0];
                sum[1] += self.points[index][1];
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some([sum[0] / count as f64, sum[1] / count as f64])
        }
    }

    fn centroid_displacement(&self, polygon: &CachedPolygon) -> Result<f64, PoseGeometryError> {
        if self.count == 0 {
            return Ok(0.0);
        }
        let mut sum = [0.0, 0.0];
        for (index, point) in self.points.iter().enumerate() {
            if self.valid[index] {
                sum[0] += point[0];
                sum[1] += point[1];
            }
        }
        finite(sum[0])?;
        finite(sum[1])?;
        let centroid = [sum[0] / self.count as f64, sum[1] / self.count as f64];
        let center = [
            (polygon.min[0] + polygon.max[0]) / 2.0,
            (polygon.min[1] + polygon.max[1]) / 2.0,
        ];
        let dx = centroid[0] - center[0];
        let dy = centroid[1] - center[1];
        let x_squared = dx * dx;
        let y_squared = dy * dy;
        let squared = x_squared + y_squared;
        finite(centroid[0])?;
        finite(centroid[1])?;
        finite(center[0])?;
        finite(center[1])?;
        finite(x_squared)?;
        finite(y_squared)?;
        finite(squared)?;
        let distance = squared.sqrt();
        finite(distance)?;
        Ok(distance / polygon.diagonal)
    }
}

fn fraction(valid: &[bool; KEYPOINTS], inside: &[bool; KEYPOINTS], indices: &[usize]) -> f64 {
    let mut total = 0usize;
    let mut included = 0usize;
    for &index in indices {
        if valid[index] {
            total += 1;
            if inside[index] {
                included += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        included as f64 / total as f64
    }
}
