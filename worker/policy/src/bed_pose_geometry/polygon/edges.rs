use super::super::PoseGeometryError;
use super::CachedPolygon;

pub(in crate::bed_pose_geometry) fn contains(
    point: [f64; 2],
    polygon: &CachedPolygon,
) -> Result<bool, PoseGeometryError> {
    let mut inside = false;
    for index in 0..polygon.vertices.len() {
        let first = polygon.vertices[index];
        let second = polygon.vertices[(index + 1) % polygon.vertices.len()];
        if (first[1] > point[1]) != (second[1] > point[1]) {
            let dx = second[0] - first[0];
            let y_offset = point[1] - first[1];
            let denominator = second[1] - first[1];
            let numerator = dx * y_offset;
            let quotient = numerator / denominator;
            let intersection = quotient + first[0];
            finite(dx)?;
            finite(y_offset)?;
            finite(denominator)?;
            finite(numerator)?;
            finite(quotient)?;
            finite(intersection)?;
            if point[0] < intersection {
                inside = !inside;
            }
        }
    }
    Ok(inside)
}

pub(in crate::bed_pose_geometry) fn edge_distance(
    point: [f64; 2],
    polygon: &CachedPolygon,
) -> Result<f64, PoseGeometryError> {
    let mut minimum = f64::INFINITY;
    for index in 0..polygon.vertices.len() {
        let start = polygon.vertices[index];
        let end = polygon.vertices[(index + 1) % polygon.vertices.len()];
        let segment = [end[0] - start[0], end[1] - start[1]];
        let x_length_squared = segment[0] * segment[0];
        let y_length_squared = segment[1] * segment[1];
        let length_squared = x_length_squared + y_length_squared;
        let offset = [point[0] - start[0], point[1] - start[1]];
        let x_dot = offset[0] * segment[0];
        let y_dot = offset[1] * segment[1];
        let dot = x_dot + y_dot;
        finite(segment[0])?;
        finite(segment[1])?;
        finite(x_length_squared)?;
        finite(y_length_squared)?;
        finite(length_squared)?;
        finite(offset[0])?;
        finite(offset[1])?;
        finite(x_dot)?;
        finite(y_dot)?;
        finite(dot)?;
        let mut t = if length_squared > 0.0 {
            dot / length_squared
        } else {
            0.0
        };
        finite(t)?;
        if t < 0.0 {
            t = 0.0;
        } else if t > 1.0 {
            t = 1.0;
        }
        let x_step = t * segment[0];
        let y_step = t * segment[1];
        let closest = [start[0] + x_step, start[1] + y_step];
        let delta = [closest[0] - point[0], closest[1] - point[1]];
        let x_squared = delta[0] * delta[0];
        let y_squared = delta[1] * delta[1];
        let squared = x_squared + y_squared;
        finite(x_step)?;
        finite(y_step)?;
        finite(closest[0])?;
        finite(closest[1])?;
        finite(delta[0])?;
        finite(delta[1])?;
        finite(x_squared)?;
        finite(y_squared)?;
        finite(squared)?;
        let distance = squared.sqrt();
        finite(distance)?;
        if distance < minimum {
            minimum = distance;
        }
    }
    finite(minimum)?;
    Ok(minimum)
}

pub(in crate::bed_pose_geometry) fn finite(value: f64) -> Result<(), PoseGeometryError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PoseGeometryError::NonFinite)
    }
}
