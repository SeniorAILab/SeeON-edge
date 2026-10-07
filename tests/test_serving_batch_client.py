from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.runner import Image, PoseRunnerResult
from worker.adapters.model.batch_input import BatchInputError
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import ModelOption, ModelRegistry
from worker.adapters.model.yolo_api import YoloModel, YoloResult
from worker.adapters.model.yolo_pose import YoloPoseRunner
from worker.domains.module_compiler import CompiledDetectionModuleRegistry
from worker.interfaces.serving import BatchServingClient, ServingClient
from worker.runtime.model_composition import SharedComponentPool, compose_shared_components
from worker.types import FramePacket


@final
class _FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = xy
        self.conf = conf


@final
class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros((len(xyxy),), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.xyxy)


@final
class _FakeResult:
    def __init__(self, marker: float) -> None:
        self.keypoints = _FakeKeypoints(
            np.full((1, 17, 2), marker, dtype=np.float64),
            np.full((1, 17), 0.9, dtype=np.float64),
        )
        self.boxes = _FakeBoxes(
            np.array([[marker, marker, marker + 1.0, marker + 1.0]], dtype=np.float64),
            np.array([0.8], dtype=np.float64),
        )
        self.masks = None


@final
class _RecordingYoloModel:
    names = {0: "person"}

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def predict(
        self,
        *,
        source: Image | Sequence[Image],
        conf: float,
        verbose: bool,
        device: str,
    ) -> Sequence[YoloResult]:
        del conf, verbose, device
        images = list(source) if isinstance(source, list) else [source]
        self.calls.append(tuple(int(np.asarray(image).mean()) for image in images))
        return [_FakeResult(float(np.asarray(image).mean())) for image in images]


def _image(value: int, *, height: int = 8, width: int = 8) -> Image:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _packet(camera_id: str, seq: int, value: int) -> FramePacket:
    image = _image(value)
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(index=seq, time_sec=float(seq), image=image),
        pts=float(seq),
        seq=seq,
        width=image.shape[1],
        height=image.shape[0],
        decode_time_ms=0.0,
    )


def _registry_with(model: YoloModel) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "pose",
        lambda **kwargs: YoloPoseRunner(model=model, **_without_path(kwargs)),
    )
    return registry


def _without_path(kwargs: dict[str, ModelOption]) -> dict[str, ModelOption]:
    return {key: value for key, value in kwargs.items() if key != "model_path"}


def _batch_client(model: YoloModel) -> BatchServingClient:
    return InProcessServingClient(_registry_with(model)).batch_serving_client


def test_batch_client_satisfies_both_serving_protocols() -> None:
    client = _batch_client(_RecordingYoloModel())

    assert isinstance(client, ServingClient)
    assert isinstance(client, BatchServingClient)


def test_composition_exposes_the_model_sharing_batch_view() -> None:
    serving = InProcessServingClient(_registry_with(_RecordingYoloModel()))
    empty_registry = CompiledDetectionModuleRegistry((), {}, {}, {})

    graph = compose_shared_components(
        empty_registry,
        module_versions={},
        serving_client=serving,
        runtime="cpu",
        device="cpu",
        flags={},
        pool=SharedComponentPool(),
    )

    assert graph.batch_serving_client is serving.batch_serving_client


def test_infer_batch_issues_one_batched_forward_for_the_whole_frame_list() -> None:
    model = _RecordingYoloModel()
    client = _batch_client(model)
    frames = tuple(_packet(f"camera-{index}", seq=index, value=index * 10) for index in range(1, 6))

    results = client.infer_batch("pose", frames, device="cpu")

    assert len(model.calls) == 1
    assert model.calls[0] == (10, 20, 30, 40, 50)
    assert len(results) == len(frames)


def test_infer_batch_result_order_matches_frame_order_against_the_real_client() -> None:
    model = _RecordingYoloModel()
    client = _batch_client(model)
    frames = (
        _packet("camera-7", seq=3, value=70),
        _packet("camera-2", seq=99, value=20),
        _packet("camera-11", seq=1, value=110),
        _packet("camera-2", seq=100, value=21),
    )

    results = client.infer_batch("pose", frames, device="cpu")

    markers = []
    for result in results:
        assert isinstance(result, PoseRunnerResult)
        markers.append(int(result.boxes[0][0]))
    assert markers == [70, 20, 110, 21]


def test_infer_batch_and_create_share_one_model_instance_per_task() -> None:
    model = _RecordingYoloModel()
    serving = InProcessServingClient(_registry_with(model))
    client = serving.batch_serving_client

    assert serving.batch_serving_client is client
    first = client.create("pose", device="cpu")
    second = client.create("pose", device="cpu")
    client.infer_batch("pose", (_packet("camera-1", seq=1, value=5),), device="cpu")

    assert first is second
    assert client.create("pose", device="cpu") is first


def test_infer_batch_of_an_empty_frame_list_calls_no_model() -> None:
    model = _RecordingYoloModel()
    client = _batch_client(model)

    assert client.infer_batch("pose", (), device="cpu") == ()
    assert model.calls == []


@pytest.mark.parametrize(
    ("bad_image", "detail"),
    [
        (np.zeros((8, 8, 3), dtype=np.float32), "dtype"),
        (np.zeros((8, 8), dtype=np.uint8), "shape"),
        (np.zeros((8, 8, 4), dtype=np.uint8), "shape"),
    ],
)
def test_infer_batch_rejects_mismatched_dtype_or_shape_loudly(
    bad_image: np.ndarray, detail: str
) -> None:
    model = _RecordingYoloModel()
    client = _batch_client(model)
    good = _packet("camera-1", seq=1, value=5)
    bad = _packet("camera-2", seq=2, value=0)
    object.__setattr__(bad.borrow_host_frame(), "image", bad_image)

    with pytest.raises(BatchInputError) as raised:
        client.infer_batch("pose", (good, bad), device="cpu")

    assert "camera-2" in str(raised.value)
    assert detail in str(raised.value)
    assert model.calls == []


def test_infer_batch_rejects_ragged_geometry_instead_of_coercing() -> None:
    model = _RecordingYoloModel()
    client = _batch_client(model)
    small = _packet("camera-1", seq=1, value=5)
    tall_image = _image(6, height=16, width=8)
    tall = FramePacket(
        camera_id="camera-9",
        frame=Frame(index=2, time_sec=2.0, image=tall_image),
        pts=2.0,
        seq=2,
        width=8,
        height=16,
        decode_time_ms=0.0,
    )

    with pytest.raises(BatchInputError) as raised:
        client.infer_batch("pose", (small, tall), device="cpu")

    assert "camera-9" in str(raised.value)
    assert model.calls == []


def test_infer_batch_rejects_a_non_batch_task() -> None:
    client = _batch_client(_RecordingYoloModel())

    with pytest.raises(BatchInputError) as raised:
        client.infer_batch("bed", (_packet("camera-1", seq=1, value=5),), device="cpu")

    assert "bed" in str(raised.value)
