from __future__ import annotations

import logging
from typing import Final

import pytest

from contracts.observation import BoundingBox
from worker.domains.bed_exit.geometry import (
    _bed_polygon_mask,
    containment_ratio,
)

DIAMOND: Final = ((50, 0), (100, 50), (50, 100), (0, 50))

NOTCHED_L: Final = ((0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100))


def _bed(polygon: tuple[tuple[int, int], ...] | None) -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=polygon)


def test_convex_polygon_person_fully_inside_is_fully_contained() -> None:
    bed = _bed(DIAMOND)
    person = BoundingBox(40, 40, 60, 60, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 1.0


def test_convex_polygon_person_outside_polygon_but_inside_aabb() -> None:
    bed = _bed(DIAMOND)
    person = BoundingBox(80, 80, 95, 95, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_non_convex_polygon_person_inside_main_body() -> None:
    bed = _bed(NOTCHED_L)
    person = BoundingBox(10, 10, 50, 50, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 1.0


def test_non_convex_polygon_person_in_the_notch_is_not_contained() -> None:
    bed = _bed(NOTCHED_L)
    person = BoundingBox(70, 70, 90, 90, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_no_polygon_falls_back_to_exact_aabb_formula() -> None:
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 2_500 / 10_000


def test_no_polygon_person_fully_outside_is_zero() -> None:
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(200, 200, 250, 250, 0.9)

    assert containment_ratio(person, bed) == 0.0


def test_degenerate_person_box_is_zero_regardless_of_polygon() -> None:
    bed = _bed(DIAMOND)
    person = BoundingBox(50, 50, 50, 80, 0.9)

    assert containment_ratio(person, bed) == 0.0


def test_person_box_fully_outside_bed_aabb_is_zero() -> None:
    bed = _bed(DIAMOND)
    person = BoundingBox(150, 150, 180, 180, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_person_box_one_pixel_outside_bed_aabb_is_zero() -> None:
    bed = _bed(DIAMOND)
    person = BoundingBox(-20, -20, 0, 0, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_self_intersecting_polygon_is_rasterized_not_rejected() -> None:
    self_intersecting: Final = ((0, 0), (100, 0), (100, 100), (0, 100), (5, -5))

    mask_info = _bed_polygon_mask(self_intersecting)

    assert mask_info is not None
    bed = _bed(self_intersecting)
    person = BoundingBox(20, 20, 80, 80, 0.9)
    assert containment_ratio(person, bed) == 1.0


def test_polygon_with_fewer_than_three_points_falls_back_to_aabb() -> None:
    degenerate: Final = ((10, 10), (90, 90))
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=degenerate)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 2_500 / 10_000
    assert _bed_polygon_mask(degenerate) is None


def test_collinear_polygon_falls_back_to_aabb() -> None:
    collinear: Final = ((0, 0), (25, 25), (50, 50), (75, 75), (100, 100))
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=collinear)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 2_500 / 10_000
    assert _bed_polygon_mask(collinear) is None


def test_unusable_polygon_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    degenerate: Final = ((11, 11), (91, 91))

    with caplog.at_level(logging.WARNING, logger="worker.domains.bed_exit.geometry"):
        result = _bed_polygon_mask(degenerate)

    assert result is None
    assert "falling back to AABB" in caplog.text
