use crate::bed_exit::BoundingBox;

use super::PoseGeometryError;

mod edges;

pub(super) use edges::{contains, edge_distance, finite};

const ZERO_AREA: f64 = 1e-9;

#[derive(Debug)]
pub(super) struct CachedPolygon {
    pub(super) bed_id: u64,
    pub(super) vertices: Vec<[f64; 2]>,
    pub(super) min: [f64; 2],
    pub(super) max: [f64; 2],
    pub(super) span: [f64; 2],
    pub(super) diagonal: f64,
}

pub(super) fn prepare(
    beds: &[BoundingBox],
    frame_size: [i64; 2],
    polygon_size: Option<[i64; 2]>,
    max_beds: usize,
    max_vertices: usize,
) -> Result<Vec<CachedPolygon>, PoseGeometryError> {
    if beds.len() > max_beds {
        return Err(PoseGeometryError::Capacity);
    }
    let total_vertices = beds.iter().try_fold(0usize, |total, bed| {
        total
            .checked_add(source_count(bed))
            .ok_or(PoseGeometryError::Capacity)
    })?;
    if total_vertices > max_vertices {
        return Err(PoseGeometryError::Capacity);
    }
    let Some(scale) = scale_for(frame_size, polygon_size)? else {
        return Ok(Vec::new());
    };
    let mut cached = Vec::new();
    cached
        .try_reserve_exact(beds.len())
        .map_err(|_| PoseGeometryError::Allocation)?;
    for (index, bed) in beds.iter().enumerate() {
        let count = source_count(bed);
        if count < 3 {
            continue;
        }
        let bed_id = u64::try_from(index).map_err(|_| PoseGeometryError::Range)?;
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact(count)
            .map_err(|_| PoseGeometryError::Allocation)?;
        match &bed.polygon {
            Some(points) => {
                for &(x, y) in points {
                    vertices.push(scaled([x as f64, y as f64], scale)?);
                }
            }
            None => {
                for point in [
                    [bed.x1, bed.y1],
                    [bed.x2, bed.y1],
                    [bed.x2, bed.y2],
                    [bed.x1, bed.y2],
                ] {
                    vertices.push(scaled([point[0] as f64, point[1] as f64], scale)?);
                }
            }
        }
        if let Some(polygon) = validate(bed_id, vertices)? {
            cached.push(polygon);
        }
    }
    Ok(cached)
}

fn source_count(bed: &BoundingBox) -> usize {
    match &bed.polygon {
        Some(points) => points.len(),
        None if bed.x2 > bed.x1 && bed.y2 > bed.y1 => 4,
        None => 0,
    }
}

fn scale_for(
    frame_size: [i64; 2],
    polygon_size: Option<[i64; 2]>,
) -> Result<Option<[f64; 2]>, PoseGeometryError> {
    let Some(source) = polygon_size else {
        return Ok(Some([1.0, 1.0]));
    };
    if source[0] <= 0 || source[1] <= 0 || frame_size[0] <= 0 || frame_size[1] <= 0 {
        return Ok(None);
    }
    let scale = [
        frame_size[0] as f64 / source[0] as f64,
        frame_size[1] as f64 / source[1] as f64,
    ];
    if scale.iter().all(|value| value.is_finite()) {
        Ok(Some(scale))
    } else {
        Err(PoseGeometryError::NonFinite)
    }
}

fn scaled(point: [f64; 2], scale: [f64; 2]) -> Result<[f64; 2], PoseGeometryError> {
    let value = [point[0] * scale[0], point[1] * scale[1]];
    finite(value[0])?;
    finite(value[1])?;
    Ok(value)
}

fn validate(
    bed_id: u64,
    vertices: Vec<[f64; 2]>,
) -> Result<Option<CachedPolygon>, PoseGeometryError> {
    let Some(first) = vertices.first().copied() else {
        return Ok(None);
    };
    let mut min = first;
    let mut max = first;
    for point in &vertices {
        for axis in 0..2 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let span = [max[0] - min[0], max[1] - min[1]];
    finite(span[0])?;
    finite(span[1])?;
    if vertices.len() < 3 || span[0] <= 0.0 || span[1] <= 0.0 || collinear(&vertices)? {
        return Ok(None);
    }
    if area(&vertices)?.abs() <= ZERO_AREA {
        return Ok(None);
    }
    let width_squared = span[0] * span[0];
    let height_squared = span[1] * span[1];
    let squared = width_squared + height_squared;
    finite(width_squared)?;
    finite(height_squared)?;
    finite(squared)?;
    let diagonal = squared.sqrt();
    finite(diagonal)?;
    Ok(Some(CachedPolygon {
        bed_id,
        vertices,
        min,
        max,
        span,
        diagonal,
    }))
}

fn collinear(vertices: &[[f64; 2]]) -> Result<bool, PoseGeometryError> {
    let origin = vertices[0];
    let mut direction = None;
    for point in &vertices[1..] {
        let candidate = [point[0] - origin[0], point[1] - origin[1]];
        finite(candidate[0])?;
        finite(candidate[1])?;
        let x_squared = candidate[0] * candidate[0];
        let y_squared = candidate[1] * candidate[1];
        let squared = x_squared + y_squared;
        finite(x_squared)?;
        finite(y_squared)?;
        finite(squared)?;
        if squared > 0.0 {
            direction = Some(candidate);
            break;
        }
    }
    let Some(direction) = direction else {
        return Ok(true);
    };
    for point in vertices {
        let x = (point[0] - origin[0]) * direction[1];
        let y = (point[1] - origin[1]) * direction[0];
        let cross = x - y;
        finite(x)?;
        finite(y)?;
        finite(cross)?;
        if cross.abs() > ZERO_AREA {
            return Ok(false);
        }
    }
    Ok(true)
}

fn area(vertices: &[[f64; 2]]) -> Result<f64, PoseGeometryError> {
    let mut xy = 0.0;
    let mut yx = 0.0;
    for index in 0..vertices.len() {
        let point = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        let first = point[0] * next[1];
        let second = point[1] * next[0];
        xy += first;
        yx += second;
        finite(first)?;
        finite(second)?;
        finite(xy)?;
        finite(yx)?;
    }
    let doubled = xy - yx;
    finite(doubled)?;
    Ok(0.5 * doubled)
}
