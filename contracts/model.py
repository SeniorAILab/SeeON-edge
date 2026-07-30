from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

from contracts.frame import Frame
from contracts.observation import FrameObservation

DEFAULT_FALL_CONFIDENCE_THRESHOLD: Final = 0.2


@runtime_checkable
class ModelModule(Protocol):
    def predict(self, frame: Frame) -> FrameObservation: ...
