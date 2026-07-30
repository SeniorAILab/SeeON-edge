"""Domain-agnostic overlay drawing primitives (perception L2).

Single source of truth for drawing extracted objects (boxes, segmentation
polygons, pose skeletons, captions) onto a frame. Callers (the edge dev MJPEG
and camera workers) decide *what* to draw and pass the colors/labels;
this module only knows *how* to draw and never imports a domain. Keeping it at
L2 lets both ``demo`` (L5) and ``worker`` consume one implementation without a
ladder violation.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import BoundingBox

Color = tuple[int, int, int]

FALL_BOX_COLOR: Final[Color] = (255, 64, 64)
DETECTION_BOX_COLOR: Final[Color] = (64, 220, 120)
BED_EMPTY_COLOR: Final[Color] = (180, 180, 180)
BED_OCCUPIED_COLOR: Final[Color] = (80, 160, 255)
BED_EXIT_COLOR: Final[Color] = (255, 180, 32)
POSE_COLOR: Final[Color] = (80, 160, 255)
CAPTION_TEXT_COLOR: Final[Color] = (16, 16, 16)
CAPTION_FONT_SCALE: Final = 0.5
CAPTION_THICKNESS: Final = 1
LABEL_FONT_SCALE: Final = 0.45
MASK_FILL_ALPHA: Final = 0.3
MIN_KEYPOINT_CONFIDENCE: Final = 0.2

POSE_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def draw_box(
    image: NDArray[np.uint8],
    box: BoundingBox,
    color: Color,
    *,
    thickness: int = 2,
) -> None:
    """Draw an axis-aligned rectangle for a detection box."""
    cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, thickness)


def draw_region(
    image: NDArray[np.uint8],
    box: BoundingBox,
    color: Color,
    *,
    fill: bool = False,
    thickness: int = 2,
) -> None:
    """Draw a region as its segmentation polygon when the box carries a mask
    contour, otherwise as an axis-aligned box. ``fill`` adds a translucent mask
    fill under the polygon outline."""
    if box.polygon:
        points = np.array(box.polygon, dtype=np.int32).reshape((-1, 1, 2))
        if fill:
            mask = image.copy()
            cv2.fillPoly(mask, [points], color)
            cv2.addWeighted(mask, MASK_FILL_ALPHA, image, 1 - MASK_FILL_ALPHA, 0, image)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        return
    draw_box(image, box, color, thickness=thickness)


def draw_caption(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: Color,
) -> None:
    """Draw a filled caption chip with contrasting text above ``y``."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, CAPTION_FONT_SCALE, CAPTION_THICKNESS
    )
    top = max(y - text_height - baseline - 2, 0)
    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 2, top + text_height + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 1, top + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_FONT_SCALE,
        CAPTION_TEXT_COLOR,
        CAPTION_THICKNESS,
        cv2.LINE_AA,
    )


def draw_label(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: Color,
    *,
    scale: float = LABEL_FONT_SCALE,
) -> None:
    """Draw a plain text label (no background chip)."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def draw_pose(
    image: NDArray[np.uint8],
    keypoints: tuple[tuple[int, int, float], ...],
    *,
    color: Color = POSE_COLOR,
    dot_radius: int = 3,
    skeleton: bool = True,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> None:
    """Draw COCO-17 keypoints as confident dots, optionally with a skeleton."""
    if skeleton:
        for start, end in POSE_EDGES:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_point = keypoints[start]
            end_point = keypoints[end]
            if start_point[2] < min_confidence or end_point[2] < min_confidence:
                continue
            cv2.line(
                image,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                color,
                2,
            )
    for x, y, confidence in keypoints:
        if confidence >= min_confidence:
            cv2.circle(image, (int(x), int(y)), dot_radius, color, -1)
