//! Live, bounded producer for the numeric bed-exit input.

use std::collections::BTreeMap;

use seeon_worker::bed_containment::{ContainmentError, PolygonFallback};
use seeon_worker::bed_exit::{AssignmentSnapshot, BedExitCapacities, BedExitInput};
use seeon_worker::bed_pose_geometry::{BedPoseGeometry, PoseGeometryError};

use super::ingest::{Frame, ObservedPose};
use crate::relay::cameras::RuntimeCamera;

mod observations;
mod regions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedInputError {
    Capacity {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
    Conversion(&'static str),
    Allocation(&'static str),
    Overflow(&'static str),
    Containment(ContainmentError),
    PoseGeometry(PoseGeometryError),
}

impl std::fmt::Display for BedInputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "bed input failure: {self:?}")
    }
}

impl std::error::Error for BedInputError {}

impl From<ContainmentError> for BedInputError {
    fn from(error: ContainmentError) -> Self {
        Self::Containment(error)
    }
}

impl From<PoseGeometryError> for BedInputError {
    fn from(error: PoseGeometryError) -> Self {
        Self::PoseGeometry(error)
    }
}

pub struct BedInputProducer {
    capacities: BedExitCapacities,
    regions: regions::Regions,
    polygon_size: Option<[i64; 2]>,
    pose_geometry: Option<([i64; 2], BedPoseGeometry)>,
}

impl BedInputProducer {
    pub fn new(
        camera: &RuntimeCamera,
        capacities: BedExitCapacities,
        mask_cells: usize,
    ) -> Result<Self, BedInputError> {
        let polygon_size = regions::polygon_image_size(camera)?;
        let regions = regions::Regions::new(camera, capacities, mask_cells)?;
        Ok(Self {
            capacities,
            regions,
            polygon_size,
            pose_geometry: None,
        })
    }

    pub fn build(
        &mut self,
        frame: &Frame,
        observed: &[ObservedPose],
        assignments: &BTreeMap<u64, AssignmentSnapshot>,
    ) -> Result<BedExitInput, BedInputError> {
        observations::build(self, frame, observed, assignments)
    }

    pub fn fallbacks(&self) -> impl Iterator<Item = (usize, PolygonFallback)> + '_ {
        self.regions
            .masks
            .iter()
            .enumerate()
            .filter_map(|(index, mask)| mask.fallback().map(|reason| (index, reason)))
    }

    pub fn allocated_cells(&self) -> usize {
        self.regions.allocated_cells
    }
}

fn capacity(field: &'static str, actual: usize, limit: usize) -> Result<(), BedInputError> {
    if actual > limit {
        Err(BedInputError::Capacity {
            field,
            limit,
            actual,
        })
    } else {
        Ok(())
    }
}

fn reserve<T>(items: &mut Vec<T>, count: usize, field: &'static str) -> Result<(), BedInputError> {
    items
        .try_reserve_exact(count)
        .map_err(|_| BedInputError::Allocation(field))
}
