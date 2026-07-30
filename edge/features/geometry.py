from __future__ import annotations

from contracts.observation import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union for two axis-aligned bounding boxes.

    Returns 0.0 when the boxes do not overlap or when either box has zero area.
    """
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def greedy_match(
    existing_boxes: tuple[BoundingBox, ...],
    boxes: tuple[BoundingBox, ...],
    min_iou: float,
) -> tuple[tuple[int, int], ...]:
    """Greedily match existing boxes to incoming boxes by IoU descending."""
    pairs: list[tuple[float, int, int]] = []
    for ti, existing_box in enumerate(existing_boxes):
        for bi, box in enumerate(boxes):
            score = iou(existing_box, box)
            if score > 0.0:
                pairs.append((score, ti, bi))

    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_track_idxs: set[int] = set()
    matched_box_idxs: set[int] = set()
    matches: list[tuple[int, int]] = []

    for score, ti, bi in pairs:
        if score < min_iou:
            break
        if ti in matched_track_idxs or bi in matched_box_idxs:
            continue
        matched_track_idxs.add(ti)
        matched_box_idxs.add(bi)
        matches.append((ti, bi))

    return tuple(matches)
