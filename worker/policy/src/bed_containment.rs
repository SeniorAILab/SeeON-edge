//! Cached bed-polygon containment using the Python worker's integer-y raster fill.
use crate::bed_exit::BoundingBox;

mod raster;

/// Per-cache construction limits; stored prefix cells include each row's leading zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterLimits {
    pub max_cells: usize,
    pub max_vertices: usize,
}

/// Refusal to represent, allocate, or exactly score the requested geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainmentError {
    VertexCapacityExceeded { requested: usize, limit: usize },
    CellCapacityExceeded { requested: usize, limit: usize },
    AllocationFailed,
    CoordinateRange,
    IntegerRange,
}

/// Python-equivalent unusable-polygon diagnostics; an absent or empty polygon has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonFallback {
    Degenerate,
    ZeroArea,
    EmptyRaster,
}

/// Immutable bed AABB and, when usable, its one-time bounded polygon raster.
#[derive(Debug)]
pub struct BedContainment {
    bed_bounds: [i64; 4],
    raster: Option<raster::Raster>,
    fallback: Option<PolygonFallback>,
}

impl BedContainment {
    pub fn new(bed: &BoundingBox, limits: RasterLimits) -> Result<Self, ContainmentError> {
        let mut cached_raster = None;
        let mut fallback = None;
        if let Some(polygon) = bed.polygon.as_deref().filter(|points| !points.is_empty()) {
            if polygon.len() > limits.max_vertices {
                return Err(ContainmentError::VertexCapacityExceeded {
                    requested: polygon.len(),
                    limit: limits.max_vertices,
                });
            }
            match raster::build(polygon, limits)? {
                raster::RasterBuild::Ready(mask) => cached_raster = Some(mask),
                raster::RasterBuild::Fallback(reason) => fallback = Some(reason),
            }
        }
        Ok(Self {
            bed_bounds: [bed.x1, bed.y1, bed.x2, bed.y2],
            raster: cached_raster,
            fallback,
        })
    }

    /// Return the person-AABB area covered by this bed, ignoring person.polygon.
    pub fn ratio(&self, person: &BoundingBox) -> Result<f64, ContainmentError> {
        let person_area = area(person.x1, person.y1, person.x2, person.y2)?;
        if person_area == 0 {
            return Ok(0.0);
        }
        check_exact_area(person_area)?;
        let intersection = match &self.raster {
            Some(mask) => mask.intersection(person)?,
            None => aabb_intersection(person, self.bed_bounds)?,
        };
        check_exact_area(intersection)?;
        Ok(intersection as f64 / person_area as f64)
    }

    /// Number of stored prefix entries, including one leading zero per raster row.
    pub fn allocated_cells(&self) -> usize {
        self.raster
            .as_ref()
            .map_or(0, raster::Raster::allocated_cells)
    }

    /// Fallback reason captured once during construction, if an explicit polygon was unusable.
    pub fn fallback(&self) -> Option<PolygonFallback> {
        self.fallback
    }
}

const MAX_EXACT_FLOAT_INTEGER: u128 = 1_u128 << 53;

fn check_exact_area(value: u128) -> Result<(), ContainmentError> {
    if value > MAX_EXACT_FLOAT_INTEGER {
        Err(ContainmentError::IntegerRange)
    } else {
        Ok(())
    }
}

fn area(x1: i64, y1: i64, x2: i64, y2: i64) -> Result<u128, ContainmentError> {
    let width = positive_span(x1, x2)?;
    let height = positive_span(y1, y2)?;
    width
        .checked_mul(height)
        .ok_or(ContainmentError::IntegerRange)
}

fn positive_span(start: i64, end: i64) -> Result<u128, ContainmentError> {
    if end <= start {
        return Ok(0);
    }
    u128::try_from(i128::from(end) - i128::from(start)).map_err(|_| ContainmentError::IntegerRange)
}

fn aabb_intersection(person: &BoundingBox, bed: [i64; 4]) -> Result<u128, ContainmentError> {
    let left = person.x1.max(bed[0]);
    let top = person.y1.max(bed[1]);
    let right = person.x2.min(bed[2]);
    let bottom = person.y2.min(bed[3]);
    if right <= left || bottom <= top {
        return Ok(0);
    }
    positive_span(left, right)?
        .checked_mul(positive_span(top, bottom)?)
        .ok_or(ContainmentError::IntegerRange)
}
