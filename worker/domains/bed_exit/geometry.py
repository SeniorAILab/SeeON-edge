from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from contracts.observation import BoundingBox

# Sample-grid resolution used to approximate containment against a bed's
# traced polygon (see `_polygon_containment_ratio`). Grid points per axis =
# _POLYGON_GRID_INTERVALS + 1.
#
# Why sampling instead of exact clipping: the recorded polygon is a
# 48-point hand trace and is not guaranteed convex -- a bed with a
# nightstand or rail notch traces a concave outline. Exact convex clipping
# (e.g. Sutherland-Hodgman) is only correct for convex clip regions and
# would silently produce wrong areas here. A general polygon-clipping
# algorithm that handles non-convex regions correctly (Weiler-Atherton) is
# easy to get subtly wrong and hard for reviewers to verify. Point-sampling
# on a fixed grid handles convex and non-convex polygons identically and its
# error is simple to bound analytically (see `_polygon_containment_ratio`),
# so it is preferred over hand-rolled clipping. No exact-polygon-
# intersection library (e.g. shapely) is in this project's dependency set;
# adding one for this alone was judged not worth a new heavy dependency.
_POLYGON_GRID_INTERVALS = 32


def best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = (
        (ratio, bed_id)
        for bed_id, ratio in enumerate(containments)
        if ratio >= min_containment
    )
    best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
    return None if best is None else best[1]


def containment_ratio(person: BoundingBox, bed: BoundingBox) -> float:
    """Fraction of `person`'s box area that lies inside `bed`.

    Uses `bed.polygon` (the operator-traced bed outline) when present: the
    bed's `x1..y2` alone is an axis-aligned bounding box, which measured
    1.57x-2.07x (mean 1.94x) the true polygon area across this site's 10
    persisted bed zones (#219), inflating containment for anyone standing
    just outside the real bed but inside its AABB. Falls back to the exact
    axis-aligned-rectangle intersection -- bit-for-bit the prior formula --
    when no polygon is recorded, so cameras with no drawn bed zone keep
    working exactly as before.
    """
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if person_area <= 0:
        return 0.0
    if bed.polygon:
        return _polygon_containment_ratio(person, bed.polygon, person_area)
    return _aabb_containment_ratio(person, bed, person_area)


def _aabb_containment_ratio(person: BoundingBox, bed: BoundingBox, person_area: int) -> float:
    left = max(person.x1, bed.x1)
    top = max(person.y1, bed.y1)
    right = min(person.x2, bed.x2)
    bottom = min(person.y2, bed.y2)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    return intersection / person_area


def _polygon_containment_ratio(
    person: BoundingBox,
    polygon: tuple[tuple[int, int], ...],
    person_area: int,
) -> float:
    """Grid-sampling estimate of intersection(person_box, polygon) / person_area.

    Error bound: sampling `person`'s box on an
    (_POLYGON_GRID_INTERVALS + 1)^2 point grid and testing each point for
    polygon containment. Within an N-interval grid (N =
    _POLYGON_GRID_INTERVALS), a single straight polygon edge crossing the
    sampled box touches at most 2N-1 of the N^2 grid cells -- a straight
    line can cross at most N-1 internal grid columns and N-1 internal grid
    rows -- so one boundary edge contributes at most ~2/N to the
    containment-ratio error. With N=32 that is ~6% per edge the person's
    box straddles; in the common case of a person standing at a single bed
    edge, absolute error is bounded by ~6%, comfortably inside the margin
    this fix creates against the ~94% mean AABB overstatement it corrects.

    Cost: fixed at (_POLYGON_GRID_INTERVALS + 1)^2 = 1,089 point-in-polygon
    tests per (person, bed) pair, done as one vectorized numpy pass,
    independent of camera resolution or zoom. This runs per person per bed
    per frame at 5fps; with a handful of tracked persons and typically one
    bed region per camera, this is low-single-digit milliseconds per frame
    -- comfortably inside the frame budget even across many cameras.
    """
    bed_xs = tuple(point[0] for point in polygon)
    bed_ys = tuple(point[1] for point in polygon)
    left = max(person.x1, min(bed_xs))
    top = max(person.y1, min(bed_ys))
    right = min(person.x2, max(bed_xs))
    bottom = min(person.y2, max(bed_ys))
    if right <= left or bottom <= top:
        return 0.0  # person box doesn't even overlap the polygon's own AABB

    n = _POLYGON_GRID_INTERVALS
    xs = np.linspace(person.x1, person.x2, n + 1)
    ys = np.linspace(person.y1, person.y2, n + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = _points_in_polygon(grid_x.ravel(), grid_y.ravel(), polygon)
    return float(np.count_nonzero(inside)) / inside.size


def _points_in_polygon(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    polygon: tuple[tuple[int, int], ...],
) -> NDArray[np.bool_]:
    """Vectorized even-odd ray-casting point-in-polygon test.

    Correct for convex and non-convex simple polygons alike (unlike
    clipping algorithms that assume a convex clip region).
    """
    poly_x = np.array([point[0] for point in polygon], dtype=np.float64)
    poly_y = np.array([point[1] for point in polygon], dtype=np.float64)
    prev_x = np.roll(poly_x, 1)
    prev_y = np.roll(poly_y, 1)

    # Broadcast points (rows) against polygon edges (columns).
    points_x = px[:, None]
    points_y = py[:, None]
    edge_yi, edge_yj = poly_y[None, :], prev_y[None, :]
    edge_xi, edge_xj = poly_x[None, :], prev_x[None, :]

    crosses = (edge_yi > points_y) != (edge_yj > points_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(edge_yj == edge_yi, 1.0, edge_yj - edge_yi)
        x_intersect = edge_xi + (points_y - edge_yi) * (edge_xj - edge_xi) / denom
    crossings = crosses & (points_x < x_intersect)
    parity: NDArray[np.bool_] = crossings.sum(axis=1) % 2 == 1
    return parity


__all__ = ["best_bed_id", "containment_ratio"]
