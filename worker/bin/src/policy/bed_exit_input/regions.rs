use seeon_worker::bed_containment::{BedContainment, RasterLimits};
use seeon_worker::bed_exit::{BedExitCapacities, BoundingBox};
use seeon_worker::bed_pose_geometry::BedPoseGeometry;

use crate::relay::cameras::RuntimeCamera;

use super::{BedInputError, capacity, reserve};

pub(super) struct Regions {
    pub(super) beds: Vec<BoundingBox>,
    pub(super) masks: Vec<BedContainment>,
    pub(super) allocated_cells: usize,
    pub(super) mask_cell_limit: usize,
}

impl Regions {
    pub(super) fn new(
        camera: &RuntimeCamera,
        capacities: BedExitCapacities,
        mask_cell_limit: usize,
    ) -> Result<Self, BedInputError> {
        let configured = &camera.bed_zone_regions;
        capacity("beds", configured.len(), capacities.beds)?;
        for region in configured {
            capacity(
                "polygon_points",
                region.polygon.len(),
                capacities.polygon_points,
            )?;
        }

        let mut beds = Vec::new();
        reserve(&mut beds, configured.len(), "bed_boxes")?;
        for region in configured {
            let Some(&(first_x, first_y)) = region.polygon.first() else {
                return Err(BedInputError::Conversion("empty_bed_polygon"));
            };
            let (first_x, first_y) = (coordinate(first_x)?, coordinate(first_y)?);
            let mut polygon = Vec::new();
            reserve(&mut polygon, region.polygon.len(), "bed_polygon")?;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (first_x, first_y, first_x, first_y);
            for &(raw_x, raw_y) in &region.polygon {
                let (x, y) = (coordinate(raw_x)?, coordinate(raw_y)?);
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                polygon.push((x, y));
            }
            beds.push(BoundingBox {
                x1: min_x,
                y1: min_y,
                x2: max_x,
                y2: max_y,
                confidence: 1.0,
                polygon: Some(polygon),
            });
        }

        let mut masks = Vec::new();
        reserve(&mut masks, beds.len(), "bed_masks")?;
        let mut allocated_cells = 0usize;
        for bed in &beds {
            let remaining = mask_cell_limit
                .checked_sub(allocated_cells)
                .ok_or(BedInputError::Overflow("mask_cells"))?;
            let mask = BedContainment::new(
                bed,
                RasterLimits {
                    max_cells: remaining,
                    max_vertices: capacities.polygon_points,
                },
            )?;
            allocated_cells = allocated_cells
                .checked_add(mask.allocated_cells())
                .ok_or(BedInputError::Overflow("mask_cells"))?;
            capacity("mask_cells", allocated_cells, mask_cell_limit)?;
            masks.push(mask);
        }
        Ok(Self {
            beds,
            masks,
            allocated_cells,
            mask_cell_limit,
        })
    }
}

pub(super) fn polygon_image_size(
    camera: &RuntimeCamera,
) -> Result<Option<[i64; 2]>, BedInputError> {
    match (camera.bed_zone_image_width, camera.bed_zone_image_height) {
        (Some(width), Some(height)) => Ok(Some([
            i64::try_from(width).map_err(|_| BedInputError::Conversion("bed_image_width"))?,
            i64::try_from(height).map_err(|_| BedInputError::Conversion("bed_image_height"))?,
        ])),
        _ => Ok(None),
    }
}

pub(super) fn pose_geometry(
    beds: &[BoundingBox],
    frame_size: [i64; 2],
    polygon_size: Option<[i64; 2]>,
    capacities: BedExitCapacities,
) -> Result<BedPoseGeometry, BedInputError> {
    BedPoseGeometry::new(
        beds,
        frame_size,
        polygon_size,
        capacities.beds,
        capacities
            .beds
            .checked_mul(capacities.polygon_points)
            .ok_or(BedInputError::Overflow("polygon_points"))?,
    )
    .map_err(BedInputError::PoseGeometry)
}

fn coordinate(value: i128) -> Result<i64, BedInputError> {
    i64::try_from(value).map_err(|_| BedInputError::Conversion("bed_coordinate"))
}
