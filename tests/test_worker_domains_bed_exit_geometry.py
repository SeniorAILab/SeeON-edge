"""Unit tests for `worker.domains.bed_exit.geometry.containment_ratio`.

Covers the polygon-containment fix for #219: a bed's `x1..y2` alone is an
axis-aligned bounding box (AABB) that can be up to ~2x the true polygon area
for a rotated or irregular bed outline. `containment_ratio` now measures
against `bed.polygon` when present, falling back to the exact AABB formula
(bit-for-bit unchanged) when it is absent.
"""

from __future__ import annotations

from typing import Final

from contracts.observation import BoundingBox
from worker.domains.bed_exit.geometry import containment_ratio

# A square rotated 45 degrees ("diamond"), centered at (50, 50), inscribed in
# the AABB [0, 100] x [0, 100]. Diamond area = 100*100/2 = 5,000; AABB area =
# 10,000 -- a 2.0x inflation, matching the worst end of the #219 measured
# range (1.57x-2.07x, mean 1.94x) for a simple convex rotated rectangle.
DIAMOND: Final = ((50, 0), (100, 50), (50, 100), (0, 50))

# An "L" shape: a 100x100 square with its top-right 40x40 corner notched out
# (e.g. tracing around a nightstand at the head of the bed). This is
# non-convex -- the notch is inside the AABB but outside the polygon.
NOTCHED_L: Final = ((0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100))


def _bed(polygon: tuple[tuple[int, int], ...] | None) -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=polygon)


def test_convex_polygon_person_fully_inside_is_fully_contained() -> None:
    # Given: a person box entirely within the diamond (all 4 corners satisfy
    # |x-50| + |y-50| <= 50).
    bed = _bed(DIAMOND)
    person = BoundingBox(40, 40, 60, 60, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 1.0


def test_convex_polygon_person_outside_polygon_but_inside_aabb() -> None:
    # Given: this is the #219 regression case -- a person box that is fully
    # outside the diamond (every point has x+y >= 160, past the diamond's
    # x+y<=150 edge) yet fully inside the bed's AABB [0,100]x[0,100]. The
    # pre-fix AABB-only formula measures this as full containment
    # (intersection == person box == 225px^2, ratio == 1.0); a person
    # standing here should NOT read as "in bed".
    bed = _bed(DIAMOND)
    person = BoundingBox(80, 80, 95, 95, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 0.0


def test_non_convex_polygon_person_inside_main_body() -> None:
    # Given: a person box well within the L-shape's main body, away from the
    # notch.
    bed = _bed(NOTCHED_L)
    person = BoundingBox(10, 10, 50, 50, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 1.0


def test_non_convex_polygon_person_in_the_notch_is_not_contained() -> None:
    # Given: a person box sitting entirely in the notched-out corner --
    # inside the L-shape's AABB, but outside the actual (non-convex)
    # polygon. A convex-only algorithm (e.g. Sutherland-Hodgman clipping)
    # would get this wrong; ray-casting point sampling does not.
    bed = _bed(NOTCHED_L)
    person = BoundingBox(70, 70, 90, 90, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 0.0


def test_no_polygon_falls_back_to_exact_aabb_formula() -> None:
    # Given: a bed with no persisted polygon (the pre-#219 shape, and still
    # the shape for any camera with no drawn bed zone) and a person box that
    # partially overlaps it.
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then: intersection is [50,100]x[50,100] = 2,500; person_area is
    # 100*100 = 10,000 -- the exact pre-fix formula, unchanged.
    assert ratio == 2_500 / 10_000


def test_no_polygon_person_fully_outside_is_zero() -> None:
    # Given
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(200, 200, 250, 250, 0.9)

    # When / Then
    assert containment_ratio(person, bed) == 0.0


def test_degenerate_person_box_is_zero_regardless_of_polygon() -> None:
    # Given: a zero-area person box (guards the person_area <= 0 short
    # circuit still applies before any polygon sampling is attempted).
    bed = _bed(DIAMOND)
    person = BoundingBox(50, 50, 50, 80, 0.9)

    # When / Then
    assert containment_ratio(person, bed) == 0.0
