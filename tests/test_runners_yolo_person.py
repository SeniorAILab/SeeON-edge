from __future__ import annotations

import numpy as np

from edge.runners.yolo_person import COCO_PERSON_CLASS_ID, YoloPersonRunner, _extract_person_boxes


class _Tensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor(((10.9, 20.1, 110.7, 220.3), (1, 2, 3, 4), (30, 40, 130, 240)))
        self.conf = _Tensor((0.83, 0.91, 0.20))
        self.cls = _Tensor((COCO_PERSON_CLASS_ID, 59, COCO_PERSON_CLASS_ID))

    def __len__(self) -> int:
        return 3


class _Result:
    boxes = _Boxes()


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self, *, source: np.ndarray, conf: float, verbose: bool, device: str
    ) -> tuple[_Result, ...]:
        self.calls += 1
        assert source.shape == (2, 2, 3)
        assert conf == 0.25
        assert verbose is False
        assert device == "mps"
        return (_Result(),)


def test_extract_person_boxes_filters_coco_person_class_and_confidence() -> None:
    assert _extract_person_boxes(_Result(), 0.25) == ((10, 20, 110, 220, 0.83),)


def test_yolo_person_runner_is_lazy_and_predicts_runner_protocol_shape() -> None:
    runner = YoloPersonRunner(model_path="/missing/yolo26n.pt", confidence=0.25, device="mps")
    model = _Model()
    runner._model = model

    output = runner.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    assert output.kind == "person"
    assert output.boxes == ((10, 20, 110, 220, 0.83),)
    assert model.calls == 1
