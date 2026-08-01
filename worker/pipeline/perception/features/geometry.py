from __future__ import annotations

from contracts.observation import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Return intersection-over-union for two axis-aligned boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
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
    """Match existing boxes to incoming boxes by descending IoU."""
    pairs: list[tuple[float, int, int]] = []
    for track_index, existing_box in enumerate(existing_boxes):
        for box_index, box in enumerate(boxes):
            score = iou(existing_box, box)
            if score > 0.0:
                pairs.append((score, track_index, box_index))
    pairs.sort(key=lambda pair: pair[0], reverse=True)

    matched_track_indices: set[int] = set()
    matched_box_indices: set[int] = set()
    matches: list[tuple[int, int]] = []
    for score, track_index, box_index in pairs:
        if score < min_iou:
            break
        if track_index in matched_track_indices or box_index in matched_box_indices:
            continue
        matched_track_indices.add(track_index)
        matched_box_indices.add(box_index)
        matches.append((track_index, box_index))
    return tuple(matches)


__all__ = ["greedy_match", "iou"]
