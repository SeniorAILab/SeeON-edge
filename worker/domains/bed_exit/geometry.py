from __future__ import annotations

from contracts.observation import BoundingBox


def best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = (
        (ratio, bed_id)
        for bed_id, ratio in enumerate(containments)
        if ratio >= min_containment
    )
    best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
    return None if best is None else best[1]


def containment_ratio(person: BoundingBox, bed: BoundingBox) -> float:
    left = max(person.x1, bed.x1)
    top = max(person.y1, bed.y1)
    right = min(person.x2, bed.x2)
    bottom = min(person.y2, bed.y2)
    intersection = max(0, right - left) * max(0, bottom - top)
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if intersection == 0 or person_area <= 0:
        return 0.0
    return intersection / person_area


__all__ = ["best_bed_id", "containment_ratio"]
