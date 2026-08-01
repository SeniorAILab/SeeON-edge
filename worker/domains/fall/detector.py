from __future__ import annotations

from typing import ClassVar

from contracts.observation import FrameObservation
from worker.domains.fall.classifier import (
    FallModelProtocol,
    FallWindowClassifier,
)
from worker.domains.fall.schema import FallEvent
from worker.types import BusinessEvent, DecisionInput


class FallEventLatch:
    enabled: ClassVar[bool] = True
    classifier: FallWindowClassifier
    camera_id: str
    facility_id: str
    event_count: int
    first_event_sec: float | None
    _previous_fall: bool

    def __init__(
        self,
        model: FallModelProtocol,
        *,
        camera_id: str,
        facility_id: str,
    ) -> None:
        self.classifier = FallWindowClassifier(model)
        self.camera_id = camera_id
        self.facility_id = facility_id
        self.event_count = 0
        self.first_event_sec = None
        self._previous_fall = False

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        observation = self.classifier.classify(input_value)
        time_sec = 0.0 if input_value.time_sec is None else input_value.time_sec
        event = self.update_event(_is_fall(observation), time_sec)
        if event is None:
            return ()
        return (
            BusinessEvent(
                domain="fall",
                event_type="fall",
                identity=event.event_count,
                camera_id=self.camera_id,
                facility_id=self.facility_id,
                time_sec=event.onset_sec,
                probability=_fall_probability(observation),
            ),
        )

    def update_signal(self, is_fall: bool, time_sec: float | None = None) -> bool:
        onset_sec = 0.0 if time_sec is None else time_sec
        onset = is_fall and not self._previous_fall
        if onset:
            self.event_count += 1
            if self.first_event_sec is None:
                self.first_event_sec = onset_sec
        self._previous_fall = is_fall
        return onset

    def update_event(self, is_fall: bool, time_sec: float) -> FallEvent | None:
        if not self.update_signal(is_fall, time_sec):
            return None
        first_event_sec = self.first_event_sec
        if first_event_sec is None:
            first_event_sec = time_sec
        return FallEvent(
            event_count=self.event_count,
            onset_sec=time_sec,
            first_event_sec=first_event_sec,
        )


def _is_fall(observation: FrameObservation) -> bool:
    return any(label.is_fall for label in observation.labels)


def _fall_probability(observation: FrameObservation) -> float:
    return max(
        (label.confidence for label in observation.labels if label.is_fall),
        default=1.0,
    )


__all__ = ["FallEventLatch"]
