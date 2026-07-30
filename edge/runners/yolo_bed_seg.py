"""YOLO bed instance-segmentation runner implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from contracts.artifacts import bed_seg_weight_path
from contracts.runner import BedRunnerResult, bed_result

# COCO class index for "bed" in the standard 80-class COCO label set. The weight
# is asserted to actually map this index to "bed" at runtime (guards against a
# weight-family rename); a mismatch degrades gracefully to "no bed".
COCO_BED_CLASS_ID: Final = 59
BED_MODEL_CONFIDENCE: Final = 0.25
BED_NMS_IOU_THRESHOLD: Final = 0.5
BED_MERGE_IOU_THRESHOLD: Final = 0.5
BED_MASK_MAX_POINTS: Final = 48


class YoloBedSegRunner:
    """Low-frequency COCO instance-segmentation runner returning bed masks (class 59).

    Replaces the bbox-only detector (issue #243): each bed is returned as
    ``(x1, y1, x2, y2, conf, polygon)`` where ``polygon`` is the mask contour in
    image pixels. The bbox is derived from the mask for the bed-exit containment
    logic; the polygon is carried through for shape-accurate rendering — a loose
    axis-aligned box wraps a fisheye-distorted bed poorly.

    Robustness: the COCO class id for "bed" is asserted against the loaded model's
    ``names`` map at construction. If the weight family does not map index 59 →
    "bed", or a frame yields no masks (detection-only weight), ``detect_beds``
    degrades gracefully to an empty tuple rather than guessing.
    """

    def __init__(
        self,
        model_path: str = str(bed_seg_weight_path()),
        confidence: float = BED_MODEL_CONFIDENCE,
        max_points: int = BED_MASK_MAX_POINTS,
        device: str = "cpu",
    ) -> None:
        self._model = _load_yolo_model(Path(model_path))
        self._confidence = confidence
        self._max_points = max_points
        self._device = device
        self._bed_class_id = _resolve_bed_class_id(getattr(self._model, "names", None))

    def detect_beds(self, frame: NDArray) -> BedRunnerResult:
        """Return tagged bed instances ``(x1,y1,x2,y2,conf,polygon)`` for class-59 masks.

        No internal cap or merge: instance masks are already separated by the
        model. Cross-frame dedup and deterministic ordering happen in
        ``BedDetector`` (issue #244: no hard cap).
        """
        if self._bed_class_id is None:
            return bed_result(())
        results = self._model.predict(
            source=frame, conf=self._confidence, verbose=False, device=self._device
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return bed_result(())
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        polygons = r.masks.xy

        return bed_result(tuple(
            (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(conf),
                _simplify_polygon(poly, self._max_points),
            )
            for box, conf, cls, poly in zip(xyxy, confs, classes, polygons, strict=True)
            if int(cls) == self._bed_class_id and float(conf) >= self._confidence
        ))

    def run(self, frame: NDArray) -> BedRunnerResult:
        return self.detect_beds(frame)


def _simplify_polygon(points: NDArray, max_points: int) -> tuple[tuple[int, int], ...]:
    """Downsample a mask contour to <= ``max_points`` integer vertices (uniform stride)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return ()
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]
    return tuple((int(round(x)), int(round(y))) for x, y in pts)


def dedupe_bed_boxes(
    boxes: tuple[tuple[int, int, int, int, float], ...],
    *,
    max_beds: int | None = None,
) -> tuple[tuple[int, int, int, int, float], ...]:
    """Apply confidence NMS, overlap merge, and deterministic ordering.

    No hard cap by default: every distinct bed survives. Pass ``max_beds`` only
    to opt into an explicit ceiling (e.g. a test fixture); ``None`` keeps all.
    """
    if not boxes:
        return ()
    if max_beds is not None and max_beds <= 0:
        return ()

    nms_boxes: list[tuple[int, int, int, int, float]] = []
    for box in sorted(boxes, key=lambda b: (-b[4], b[0], b[1], b[2], b[3])):
        if all(_box_iou(box, kept) < BED_NMS_IOU_THRESHOLD for kept in nms_boxes):
            nms_boxes.append(box)

    merged = _merge_overlapping_beds(tuple(nms_boxes))
    ordered = sorted(merged, key=lambda b: (b[0], b[1], b[2], b[3], -b[4]))
    return tuple(ordered if max_beds is None else ordered[:max_beds])


def _merge_overlapping_beds(
    boxes: tuple[tuple[int, int, int, int, float], ...],
) -> tuple[tuple[int, int, int, int, float], ...]:
    clusters: list[list[tuple[int, int, int, int, float]]] = []
    for box in sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3], -b[4])):
        for cluster in clusters:
            if any(_box_iou(box, existing) >= BED_MERGE_IOU_THRESHOLD for existing in cluster):
                cluster.append(box)
                break
        else:
            clusters.append([box])

    merged: list[tuple[int, int, int, int, float]] = []
    for cluster in clusters:
        x1 = min(box[0] for box in cluster)
        y1 = min(box[1] for box in cluster)
        x2 = max(box[2] for box in cluster)
        y2 = max(box[3] for box in cluster)
        confidence = max(box[4] for box in cluster)
        merged.append((x1, y1, x2, y2, confidence))
    return tuple(merged)


def _box_iou(
    a: tuple[int, int, int, int, float],
    b: tuple[int, int, int, int, float],
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    width = max(0, right - left)
    height = max(0, bottom - top)
    intersection = width * height
    if intersection == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _resolve_bed_class_id(names: object) -> int | None:
    """Return the class id mapping to 'bed', preferring the COCO index 59.

    Returns ``None`` when the model's ``names`` map does not expose a "bed"
    class, so callers degrade to the graceful no-bed state.
    """
    if not isinstance(names, dict):
        return None
    if str(names.get(COCO_BED_CLASS_ID, "")).lower() == "bed":
        return COCO_BED_CLASS_ID
    for class_id, class_name in names.items():
        if str(class_name).lower() == "bed":
            return int(class_id)
    return None


def _load_yolo_model(weight_path: Path | str):
    from ultralytics import YOLO

    return YOLO(str(weight_path))
