from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from contracts.observation import DetectionResult

Image: TypeAlias = NDArray[np.uint8]
PoseOutput: TypeAlias = Sequence[Sequence[float]]
BoxOutput: TypeAlias = Sequence[Sequence[float]]
# One bed-instance row: (x1, y1, x2, y2, confidence) with an OPTIONAL 6th element
# carrying the mask-contour polygon as a sequence of (x, y) points. The optional
# polygon element is why this is NOT simply Sequence[float]; `_bed_box_from_output`
# consumes both the 5-tuple and the 6-tuple (issue #243 bed instance segmentation).
BedBoxPolygon: TypeAlias = Sequence[Sequence[int]]
BedBoxOutput: TypeAlias = Sequence[float | BedBoxPolygon]
RunnerResultKind: TypeAlias = Literal["pose", "person", "bed", "detection"]


@dataclass(frozen=True, slots=True)
class PoseRunnerResult:
    kind: Literal["pose"]
    poses: PoseOutput
    boxes: BoxOutput


@dataclass(frozen=True, slots=True)
class PersonRunnerResult:
    kind: Literal["person"]
    boxes: BoxOutput


@dataclass(frozen=True, slots=True)
class BedRunnerResult:
    kind: Literal["bed"]
    boxes: Iterable[BedBoxOutput]


@dataclass(frozen=True, slots=True)
class DetectionRunnerResult:
    kind: Literal["detection"]
    detections: DetectionResult


RunnerResult: TypeAlias = (
    PoseRunnerResult | PersonRunnerResult | BedRunnerResult | DetectionRunnerResult
)
RunnerOutput: TypeAlias = RunnerResult


def pose_result(poses: PoseOutput, boxes: BoxOutput) -> PoseRunnerResult:
    return PoseRunnerResult(kind="pose", poses=poses, boxes=boxes)


def person_result(boxes: BoxOutput) -> PersonRunnerResult:
    return PersonRunnerResult(kind="person", boxes=boxes)


def bed_result(boxes: Iterable[BedBoxOutput]) -> BedRunnerResult:
    return BedRunnerResult(kind="bed", boxes=boxes)


def detection_result(detections: DetectionResult) -> DetectionRunnerResult:
    return DetectionRunnerResult(kind="detection", detections=detections)


class RunRunnerProtocol(Protocol):
    def run(self, image: Image) -> RunnerResult: ...


# The camera per-frame runner seam: `CameraWorker._run_runner` invokes a runner
# through `run(image)` or a plain callable only. Fall models are NOT camera
# runners — they expose `predict(features) -> float` and are consumed by
# `FallWindowClassifier`, so they intentionally do not implement this protocol.
RunnerProtocol: TypeAlias = RunRunnerProtocol | Callable[[Image], RunnerResult]


__all__ = [
    "BedBoxOutput",
    "BedBoxPolygon",
    "BedRunnerResult",
    "BoxOutput",
    "DetectionRunnerResult",
    "Image",
    "PersonRunnerResult",
    "PoseOutput",
    "PoseRunnerResult",
    "RunRunnerProtocol",
    "RunnerOutput",
    "RunnerProtocol",
    "RunnerResult",
    "RunnerResultKind",
    "bed_result",
    "detection_result",
    "person_result",
    "pose_result",
]
