from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from contracts.observation import BoundingBox

LOGGER: Final = logging.getLogger(__name__)

_MASK_CACHE_SIZE: Final = 64


def best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = (
        (ratio, bed_id) for bed_id, ratio in enumerate(containments) if ratio >= min_containment
    )
    best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
    return None if best is None else best[1]


def containment_ratio(person: BoundingBox, bed: BoundingBox) -> float:
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if person_area <= 0:
        return 0.0
    if bed.polygon:
        mask_info = _bed_polygon_mask(bed.polygon)
        if mask_info is not None:
            return _mask_containment_ratio(person, mask_info, person_area)
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


@dataclass(frozen=True, slots=True)
class _BedMask:
    origin_x: int
    origin_y: int
    width: int
    height: int
    row_prefix_sums: tuple[tuple[int, ...], ...]


@lru_cache(maxsize=_MASK_CACHE_SIZE)
def _bed_polygon_mask(polygon: tuple[tuple[int, int], ...]) -> _BedMask | None:
    if len(polygon) < 3 or _all_collinear(polygon):
        LOGGER.warning(
            "bed polygon has fewer than 3 points or is degenerate (all "
            "points collinear); falling back to AABB containment for this "
            "bed region: %r",
            polygon,
        )
        return None

    xs = tuple(point[0] for point in polygon)
    ys = tuple(point[1] for point in polygon)
    origin_x, origin_y = min(xs), min(ys)
    width = max(xs) - origin_x
    height = max(ys) - origin_y
    if width <= 0 or height <= 0:
        LOGGER.warning(
            "bed polygon has zero-area bounding box; falling back to AABB "
            "containment for this bed region: %r",
            polygon,
        )
        return None

    row_prefix_sums = _rasterize_rows(polygon, origin_x, origin_y, width, height)
    if not any(row[-1] > 0 for row in row_prefix_sums):
        LOGGER.warning(
            "bed polygon rasterized to an empty mask; falling back to AABB "
            "containment for this bed region: %r",
            polygon,
        )
        return None
    return _BedMask(origin_x, origin_y, width, height, row_prefix_sums)


def _rasterize_rows(
    polygon: tuple[tuple[int, int], ...],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, ...], ...]:
    shifted = tuple((point[0] - origin_x, point[1] - origin_y) for point in polygon)
    edge_count = len(shifted)
    edges = tuple(
        (shifted[index], shifted[(index + 1) % edge_count]) for index in range(edge_count)
    )
    rows: list[tuple[int, ...]] = []
    for y in range(height):
        intersections: list[float] = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                continue
            low_y, high_y = (y0, y1) if y0 < y1 else (y1, y0)
            if not (low_y <= y < high_y):
                continue
            t = (y - y0) / (y1 - y0)
            intersections.append(x0 + t * (x1 - x0))
        intersections.sort()
        delta = [0] * (width + 1)
        for pair_index in range(0, len(intersections) - 1, 2):
            start_column = max(math.ceil(intersections[pair_index] - 0.5), 0)
            end_column = min(math.ceil(intersections[pair_index + 1] - 0.5), width)
            if end_column > start_column:
                delta[start_column] += 1
                delta[end_column] -= 1
        cumulative = [0] * (width + 1)
        running = 0
        for column in range(width):
            running += delta[column]
            cumulative[column + 1] = cumulative[column] + (1 if running > 0 else 0)
        rows.append(tuple(cumulative))
    return tuple(rows)


def _all_collinear(points: tuple[tuple[int, int], ...]) -> bool:
    origin_x, origin_y = points[0]
    direction_x, direction_y = 0, 0
    for point_x, point_y in points[1:]:
        direction_x, direction_y = point_x - origin_x, point_y - origin_y
        if direction_x != 0 or direction_y != 0:
            break
    else:
        return True
    return all(
        (point_x - origin_x) * direction_y - (point_y - origin_y) * direction_x == 0
        for point_x, point_y in points
    )


def _mask_containment_ratio(person: BoundingBox, mask_info: _BedMask, person_area: int) -> float:
    left = max(person.x1 - mask_info.origin_x, 0)
    top = max(person.y1 - mask_info.origin_y, 0)
    right = min(person.x2 - mask_info.origin_x, mask_info.width)
    bottom = min(person.y2 - mask_info.origin_y, mask_info.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = sum(
        mask_info.row_prefix_sums[row][right] - mask_info.row_prefix_sums[row][left]
        for row in range(top, bottom)
    )
    if intersection == 0:
        return 0.0
    return intersection / person_area


__all__ = ["best_bed_id", "containment_ratio"]
