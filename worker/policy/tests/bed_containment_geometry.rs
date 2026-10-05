use seeon_worker::bed_containment::{
    BedContainment, ContainmentError, PolygonFallback, RasterLimits,
};
use seeon_worker::bed_exit::BoundingBox;

fn bounds(x1: i64, y1: i64, x2: i64, y2: i64, polygon: Option<Vec<(i64, i64)>>) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        confidence: 1.0,
        polygon,
    }
}

fn limits() -> RasterLimits {
    RasterLimits {
        max_cells: 256,
        max_vertices: 64,
    }
}

fn rectangle() -> BoundingBox {
    bounds(0, 0, 4, 4, Some(vec![(0, 0), (4, 0), (4, 4), (0, 4)]))
}

#[test]
fn rectangle_partial_outside_and_nonpositive_person_area() {
    let cache = BedContainment::new(&rectangle(), limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    assert_eq!(cache.allocated_cells(), 20);
    assert_eq!(cache.fallback(), None);
    assert_eq!(cache.ratio(&bounds(1, 1, 3, 3, None)), Ok(1.0));
    assert_eq!(cache.ratio(&bounds(-1, -1, 1, 1, None)), Ok(0.25));
    assert_eq!(cache.ratio(&bounds(5, 0, 6, 1, None)), Ok(0.0));
    assert_eq!(cache.ratio(&bounds(2, 1, 1, 3, None)), Ok(0.0));
    assert_eq!(cache.ratio(&bounds(1, 1, 3, 1, None)), Ok(0.0));
}

#[test]
fn concave_polygon_does_not_substitute_its_aabb() {
    let bed = bounds(
        0,
        0,
        4,
        4,
        Some(vec![(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]),
    );
    let cache = BedContainment::new(&bed, limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    assert_eq!(cache.ratio(&bounds(2, 2, 4, 4, None)), Ok(0.0));
    assert_eq!(cache.fallback(), None);
}

#[test]
fn self_crossing_polygon_is_filled_by_even_odd_rule() {
    let bed = bounds(0, 0, 4, 4, Some(vec![(0, 0), (4, 4), (0, 4), (4, 0)]));
    let cache = BedContainment::new(&bed, limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    assert_eq!(cache.ratio(&bounds(1, 1, 3, 2, None)), Ok(1.0));
    assert_eq!(cache.fallback(), None);
}

#[test]
fn degenerate_empty_raster_and_empty_polygon_have_distinct_fallbacks() {
    let line = bounds(0, 0, 4, 4, Some(vec![(0, 0), (2, 2), (4, 4)]));
    let line_cache = BedContainment::new(&line, limits());
    assert!(matches!(line_cache, Ok(_)));
    let Ok(line_cache) = line_cache else { return };
    assert_eq!(line_cache.fallback(), Some(PolygonFallback::Degenerate));
    assert_eq!(line_cache.ratio(&bounds(1, 1, 3, 3, None)), Ok(1.0));
    assert_eq!(line_cache.allocated_cells(), 0);

    let empty_mask = bounds(0, 0, 1, 2, Some(vec![(0, 0), (1, 2), (0, 2)]));
    let empty_cache = BedContainment::new(&empty_mask, limits());
    assert!(matches!(empty_cache, Ok(_)));
    let Ok(empty_cache) = empty_cache else { return };
    assert_eq!(empty_cache.fallback(), Some(PolygonFallback::EmptyRaster));
    assert_eq!(empty_cache.ratio(&bounds(0, 0, 1, 1, None)), Ok(1.0));
    assert_eq!(empty_cache.allocated_cells(), 0);

    let no_polygon = bounds(0, 0, 4, 4, Some(Vec::new()));
    let no_polygon_cache = BedContainment::new(&no_polygon, limits());
    assert!(matches!(no_polygon_cache, Ok(_)));
    let Ok(no_polygon_cache) = no_polygon_cache else {
        return;
    };
    assert_eq!(no_polygon_cache.fallback(), None);
    assert_eq!(no_polygon_cache.allocated_cells(), 0);
}

#[test]
fn scanline_uses_integer_y_not_y_plus_half() {
    let bed = bounds(0, 0, 3, 3, Some(vec![(0, 0), (3, 2), (3, 3), (0, 3)]));
    let cache = BedContainment::new(&bed, limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    assert_eq!(cache.ratio(&bounds(0, 1, 1, 2, None)), Ok(1.0));
    assert_eq!(cache.ratio(&bounds(1, 1, 2, 2, None)), Ok(0.0));
}

#[test]
fn capacity_refusal_is_not_a_polygon_fallback() {
    let polygon = rectangle();
    assert!(matches!(
        BedContainment::new(
            &polygon,
            RasterLimits {
                max_cells: 256,
                max_vertices: 3
            }
        ),
        Err(ContainmentError::VertexCapacityExceeded {
            requested: 4,
            limit: 3
        })
    ));
    assert!(matches!(
        BedContainment::new(
            &polygon,
            RasterLimits {
                max_cells: 19,
                max_vertices: 4
            }
        ),
        Err(ContainmentError::CellCapacityExceeded {
            requested: 20,
            limit: 19
        })
    ));
}

#[test]
fn absent_polygon_uses_bed_aabb_and_person_aabb_even_if_person_has_polygon() {
    let bed = bounds(10, 10, 20, 20, None);
    let cache = BedContainment::new(&bed, limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    let person = bounds(
        5,
        10,
        15,
        20,
        Some(vec![(100, 100), (101, 100), (100, 101)]),
    );
    assert_eq!(cache.ratio(&person), Ok(0.5));
    assert_eq!(cache.fallback(), None);
}

#[test]
fn area_beyond_exact_f64_integer_range_is_refused() {
    let cache = BedContainment::new(&bounds(0, 0, 1, 1, None), limits());
    assert!(matches!(cache, Ok(_)));
    let Ok(cache) = cache else { return };
    let edge = 1_i64 << 27;
    assert_eq!(
        cache.ratio(&bounds(0, 0, edge, edge, None)),
        Err(ContainmentError::IntegerRange)
    );
}
