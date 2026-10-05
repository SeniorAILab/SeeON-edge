use super::{ContainmentError, PolygonFallback, RasterLimits};
use crate::bed_exit::BoundingBox;

mod geometry;

pub(super) enum RasterBuild {
    Ready(Raster),
    Fallback(PolygonFallback),
}

#[derive(Debug)]
pub(super) struct Raster {
    origin_x: i64,
    origin_y: i64,
    width: usize,
    height: usize,
    prefixes: Vec<usize>,
}

impl Raster {
    pub(super) fn allocated_cells(&self) -> usize {
        self.prefixes.len()
    }

    pub(super) fn intersection(&self, person: &BoundingBox) -> Result<u128, ContainmentError> {
        let width = i128::try_from(self.width).map_err(|_| ContainmentError::CoordinateRange)?;
        let height = i128::try_from(self.height).map_err(|_| ContainmentError::CoordinateRange)?;
        let left = (i128::from(person.x1) - i128::from(self.origin_x)).max(0);
        let top = (i128::from(person.y1) - i128::from(self.origin_y)).max(0);
        let right = (i128::from(person.x2) - i128::from(self.origin_x)).min(width);
        let bottom = (i128::from(person.y2) - i128::from(self.origin_y)).min(height);
        if right <= left || bottom <= top {
            return Ok(0);
        }
        let left = usize::try_from(left).map_err(|_| ContainmentError::CoordinateRange)?;
        let top = usize::try_from(top).map_err(|_| ContainmentError::CoordinateRange)?;
        let right = usize::try_from(right).map_err(|_| ContainmentError::CoordinateRange)?;
        let bottom = usize::try_from(bottom).map_err(|_| ContainmentError::CoordinateRange)?;
        let stride = self
            .width
            .checked_add(1)
            .ok_or(ContainmentError::IntegerRange)?;
        let mut total = 0_u128;
        for row in top..bottom {
            let start = row
                .checked_mul(stride)
                .ok_or(ContainmentError::IntegerRange)?;
            let end = start
                .checked_add(stride)
                .ok_or(ContainmentError::IntegerRange)?;
            let prefix = self
                .prefixes
                .get(start..end)
                .ok_or(ContainmentError::IntegerRange)?;
            let left_count = *prefix.get(left).ok_or(ContainmentError::IntegerRange)?;
            let right_count = *prefix.get(right).ok_or(ContainmentError::IntegerRange)?;
            let row_count = right_count
                .checked_sub(left_count)
                .ok_or(ContainmentError::IntegerRange)?;
            total = total
                .checked_add(u128::try_from(row_count).map_err(|_| ContainmentError::IntegerRange)?)
                .ok_or(ContainmentError::IntegerRange)?;
        }
        Ok(total)
    }
}

pub(super) fn build(
    polygon: &[(i64, i64)],
    limits: RasterLimits,
) -> Result<RasterBuild, ContainmentError> {
    if polygon.len() < 3 || geometry::all_collinear(polygon)? {
        return Ok(RasterBuild::Fallback(PolygonFallback::Degenerate));
    }
    let Some(&(first_x, first_y)) = polygon.first() else {
        return Ok(RasterBuild::Fallback(PolygonFallback::Degenerate));
    };
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (first_x, first_y, first_x, first_y);
    for &(x, y) in polygon.iter().skip(1) {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let width_i128 = i128::from(max_x) - i128::from(min_x);
    let height_i128 = i128::from(max_y) - i128::from(min_y);
    if width_i128 <= 0 || height_i128 <= 0 {
        return Ok(RasterBuild::Fallback(PolygonFallback::ZeroArea));
    }
    let width = usize::try_from(width_i128).map_err(|_| ContainmentError::CoordinateRange)?;
    let height = usize::try_from(height_i128).map_err(|_| ContainmentError::CoordinateRange)?;
    let stride = width
        .checked_add(1)
        .ok_or(ContainmentError::CoordinateRange)?;
    let cells = stride
        .checked_mul(height)
        .ok_or(ContainmentError::IntegerRange)?;
    if cells > limits.max_cells {
        return Err(ContainmentError::CellCapacityExceeded {
            requested: cells,
            limit: limits.max_cells,
        });
    }
    if u128::try_from(width).map_err(|_| ContainmentError::CoordinateRange)? > (1_u128 << 53)
        || u128::try_from(height).map_err(|_| ContainmentError::CoordinateRange)? > (1_u128 << 53)
    {
        return Err(ContainmentError::CoordinateRange);
    }
    let mut prefixes = Vec::new();
    prefixes
        .try_reserve_exact(cells)
        .map_err(|_| ContainmentError::AllocationFailed)?;
    prefixes.resize(cells, 0);
    let mut intersections = Vec::new();
    intersections
        .try_reserve_exact(polygon.len())
        .map_err(|_| ContainmentError::AllocationFailed)?;
    let mut has_filled_cell = false;
    for row in 0..height {
        intersections.clear();
        let scan_y = i128::try_from(row).map_err(|_| ContainmentError::CoordinateRange)?;
        for edge in polygon.windows(2) {
            let Some(&point0) = edge.first() else {
                return Err(ContainmentError::CoordinateRange);
            };
            let Some(&point1) = edge.get(1) else {
                return Err(ContainmentError::CoordinateRange);
            };
            geometry::push_intersection(point0, point1, min_x, min_y, scan_y, &mut intersections)?;
        }
        let Some(&last) = polygon.last() else {
            return Err(ContainmentError::CoordinateRange);
        };
        geometry::push_intersection(
            last,
            (first_x, first_y),
            min_x,
            min_y,
            scan_y,
            &mut intersections,
        )?;
        intersections.sort_unstable_by(f64::total_cmp);
        let start = row
            .checked_mul(stride)
            .ok_or(ContainmentError::IntegerRange)?;
        let end = start
            .checked_add(stride)
            .ok_or(ContainmentError::IntegerRange)?;
        let row_prefix = prefixes
            .get_mut(start..end)
            .ok_or(ContainmentError::IntegerRange)?;
        let Some((first_cell, columns)) = row_prefix.split_first_mut() else {
            return Err(ContainmentError::IntegerRange);
        };
        *first_cell = 0;
        let mut running = 0_usize;
        let mut pair_start = 0_usize;
        for (column, cell) in columns.iter_mut().enumerate() {
            let mut filled = false;
            while pair_start < intersections.len() {
                let pair_end = pair_start
                    .checked_add(1)
                    .ok_or(ContainmentError::IntegerRange)?;
                if pair_end >= intersections.len() {
                    break;
                }
                let left = *intersections
                    .get(pair_start)
                    .ok_or(ContainmentError::IntegerRange)?;
                let right = *intersections
                    .get(pair_end)
                    .ok_or(ContainmentError::IntegerRange)?;
                let run_start = geometry::column_boundary(left, width)?;
                let run_end = geometry::column_boundary(right, width)?;
                if run_end <= run_start || column >= run_end {
                    pair_start = pair_start
                        .checked_add(2)
                        .ok_or(ContainmentError::IntegerRange)?;
                    continue;
                }
                if column >= run_start {
                    filled = true;
                }
                break;
            }
            running = running
                .checked_add(usize::from(filled))
                .ok_or(ContainmentError::IntegerRange)?;
            *cell = running;
        }
        has_filled_cell |= running > 0;
    }
    if !has_filled_cell {
        return Ok(RasterBuild::Fallback(PolygonFallback::EmptyRaster));
    }
    Ok(RasterBuild::Ready(Raster {
        origin_x: min_x,
        origin_y: min_y,
        width,
        height,
        prefixes,
    }))
}
