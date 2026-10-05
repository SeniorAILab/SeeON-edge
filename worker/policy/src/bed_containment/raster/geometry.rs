use super::super::ContainmentError;

pub(super) fn push_intersection(
    point0: (i64, i64),
    point1: (i64, i64),
    origin_x: i64,
    origin_y: i64,
    scan_y: i128,
    intersections: &mut Vec<f64>,
) -> Result<(), ContainmentError> {
    let x0 = i128::from(point0.0) - i128::from(origin_x);
    let y0 = i128::from(point0.1) - i128::from(origin_y);
    let x1 = i128::from(point1.0) - i128::from(origin_x);
    let y1 = i128::from(point1.1) - i128::from(origin_y);
    if y0 == y1 {
        return Ok(());
    }
    let low_y = y0.min(y1);
    let high_y = y0.max(y1);
    if scan_y < low_y || scan_y >= high_y {
        return Ok(());
    }
    let numerator = scan_y - y0;
    let denominator = y1 - y0;
    let t = numerator as f64 / denominator as f64;
    let intersection = x0 as f64 + t * (x1 - x0) as f64;
    if !intersection.is_finite() {
        return Err(ContainmentError::IntegerRange);
    }
    intersections.push(intersection);
    Ok(())
}

pub(super) fn column_boundary(intersection: f64, width: usize) -> Result<usize, ContainmentError> {
    let rounded = (intersection - 0.5).ceil();
    if !rounded.is_finite() {
        return Err(ContainmentError::IntegerRange);
    }
    if rounded <= 0.0 {
        return Ok(0);
    }
    if rounded >= width as f64 {
        return Ok(width);
    }
    Ok(rounded as usize)
}

pub(super) fn all_collinear(points: &[(i64, i64)]) -> Result<bool, ContainmentError> {
    let Some(&(origin_x, origin_y)) = points.first() else {
        return Ok(true);
    };
    let mut direction = None;
    for &(x, y) in points.iter().skip(1) {
        let dx = i128::from(x) - i128::from(origin_x);
        let dy = i128::from(y) - i128::from(origin_y);
        if dx != 0 || dy != 0 {
            direction = Some((dx, dy));
            break;
        }
    }
    let Some((direction_x, direction_y)) = direction else {
        return Ok(true);
    };
    for &(x, y) in points {
        let px = i128::from(x) - i128::from(origin_x);
        let py = i128::from(y) - i128::from(origin_y);
        if !products_equal(px, direction_y, py, direction_x)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn products_equal(a: i128, b: i128, c: i128, d: i128) -> Result<bool, ContainmentError> {
    let first = signed_product(a, b)?;
    let second = signed_product(c, d)?;
    Ok(first == second)
}

fn signed_product(a: i128, b: i128) -> Result<(bool, u128), ContainmentError> {
    let magnitude = a
        .unsigned_abs()
        .checked_mul(b.unsigned_abs())
        .ok_or(ContainmentError::IntegerRange)?;
    Ok(((a < 0) ^ (b < 0) && magnitude != 0, magnitude))
}
