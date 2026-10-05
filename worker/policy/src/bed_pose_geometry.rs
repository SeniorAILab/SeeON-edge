//! Cached, bounded port of Python's bed-relative COCO-17 pose geometry.
//!
//! Frame and optional polygon sizes are `[width, height]`. Polygons are
//! already in frame coordinates when `polygon_size` is `None`; when present,
//! both axes are independently scaled into `frame_size`. As in Python, a
//! nonpositive size disables all polygons only when scaling is requested.
//! Explicit polygons with fewer than three vertices are invalid and never
//! fall back to their bounding rectangle. `max_beds` and `max_vertices` bound
//! input/cache allocations before polygon coordinates are copied. Features
//! inspect only the first 17 keypoints, require finite coordinates and
//! confidence values there, and count confidence >= 0.5 as observable.

use crate::bed_exit::{BedPoseFeatures, BoundingBox};

mod points;
mod polygon;

use polygon::CachedPolygon;

/// Capacity, allocation, non-finite arithmetic, or index-range refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoseGeometryError {
    Capacity,
    Allocation,
    NonFinite,
    Range,
}

impl std::fmt::Display for PoseGeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Capacity => "pose geometry capacity exceeded",
            Self::Allocation => "pose geometry allocation failed",
            Self::NonFinite => "pose geometry input or result is non-finite",
            Self::Range => "pose geometry index is out of range",
        })
    }
}

impl std::error::Error for PoseGeometryError {}

/// Validated frame-coordinate polygons retaining their original bed indices.
#[derive(Debug)]
pub struct BedPoseGeometry {
    polygons: Vec<CachedPolygon>,
}

impl BedPoseGeometry {
    /// Validate and cache bed polygons under explicit bed and vertex budgets.
    pub fn new(
        beds: &[BoundingBox],
        frame_size: [i64; 2],
        polygon_size: Option<[i64; 2]>,
        max_beds: usize,
        max_vertices: usize,
    ) -> Result<Self, PoseGeometryError> {
        let polygons = polygon::prepare(beds, frame_size, polygon_size, max_beds, max_vertices)?;
        Ok(Self { polygons })
    }

    /// Compute Python-equivalent scalars; no valid cached bed yields missing-bed features.
    pub fn features(
        &self,
        track_id: u64,
        pose: &[[f64; 3]],
    ) -> Result<BedPoseFeatures, PoseGeometryError> {
        let person = points::Person::new(pose)?;
        let mut best: Option<BedPoseFeatures> = None;
        for polygon in &self.polygons {
            let candidate = person.against(track_id, polygon)?;
            let replace = best.is_none_or(|current| {
                candidate.torso_in_frac > current.torso_in_frac
                    || (candidate.torso_in_frac == current.torso_in_frac
                        && (candidate.keypoint_in_frac > current.keypoint_in_frac
                            || (candidate.keypoint_in_frac == current.keypoint_in_frac
                                && (candidate.hip_depth > current.hip_depth
                                    || (candidate.hip_depth == current.hip_depth
                                        && candidate.bed_id < current.bed_id)))))
            });
            if replace {
                best = Some(candidate);
            }
        }
        match best {
            Some(features) => Ok(features),
            None => Ok(person.missing_bed(track_id)),
        }
    }
}
