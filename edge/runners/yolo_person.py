"""YOLO person detection runner implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypeAlias

from numpy.typing import NDArray

from contracts.runner import PersonRunnerResult, person_result

PERSON_WEIGHTS_DIR: Final = Path(__file__).resolve().parent.parent / "models" / "person"
PERSON_MODEL_FILENAME: Final = "yolo26n.pt"
PERSON_MODEL_CONFIDENCE: Final = 0.25
COCO_PERSON_CLASS_ID: Final = 0

PersonBoxes: TypeAlias = tuple[tuple[int, int, int, int, float], ...]


class YoloPersonRunner:
    """YOLO detection runner returning COCO person boxes (class 0)."""

    def __init__(
        self,
        model_path: str = str(PERSON_WEIGHTS_DIR / PERSON_MODEL_FILENAME),
        confidence: float = PERSON_MODEL_CONFIDENCE,
        device: str = "cpu",
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence = confidence
        self._device = device
        self._model = None

    def predict(self, frame: NDArray) -> PersonRunnerResult:
        """Return tagged person boxes as ``(x1,y1,x2,y2,conf)`` tuples."""
        results = self._get_model().predict(
            source=frame, conf=self._confidence, verbose=False, device=self._device
        )
        return person_result(_extract_person_boxes(results[0], self._confidence))

    def run(self, frame: NDArray) -> PersonRunnerResult:
        return self.predict(frame)

    def _get_model(self):
        if self._model is None:
            self._model = _load_yolo_model(self._model_path)
        return self._model


def _extract_person_boxes(result: object, confidence: float) -> PersonBoxes:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    return tuple(
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf))
        for box, conf, cls in zip(xyxy, confs, classes, strict=True)
        if int(cls) == COCO_PERSON_CLASS_ID and float(conf) >= confidence
    )


def _load_yolo_model(weight_path: Path | str):
    from ultralytics import YOLO

    return YOLO(str(weight_path))
