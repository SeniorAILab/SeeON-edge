from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from contracts.event import MutableEventPayload
from contracts.observation import FrameObservation
from edge.domains.base import DomainDetector
from edge.domains.fall.schema import FallEvent


class FallInput(Protocol):
    """Prepared observation supplied by the shared perception pipeline."""

    observation: FrameObservation
    time_sec: float | None


class FallEventLatch(DomainDetector):
    """Own fall event latching for already-classified observations."""

    enabled = True

    def __init__(
        self,
        model: object | None = None,
        *,
        prepare_input: Callable[[FallInput], FrameObservation] | None = None,
    ) -> None:
        self.event_count: int = 0
        self.first_event_sec: float | None = None
        self._prev_fall: bool = False
        self._model = model
        self._prepare_input = prepare_input

    def update(self, domain_input: FallInput) -> tuple[MutableEventPayload, ...]:
        observation = domain_input.observation
        event = self.update_event(
            _observation_is_fall(observation),
            0.0 if domain_input.time_sec is None else domain_input.time_sec,
        )
        if event is None:
            return ()
        return (
            {
                "domain": "fall",
                "event_type": "fall",
                "identity": event.event_count,
                "probability": _observation_fall_probability(observation),
                "time_sec": event.onset_sec,
            },
        )

    def audit_metadata(self) -> Mapping[str, object]:
        return {
            "model_version": getattr(self._model, "version", None),
            "operating_threshold": getattr(self._model, "operating_threshold", None),
        }
    def prepare_input(self, domain_input: FallInput) -> FrameObservation:
        if self._prepare_input is None:
            return domain_input.observation
        return self._prepare_input(domain_input)

    def update_signal(self, is_fall: bool, time_sec: float | None = None) -> bool:
        if time_sec is None:
            time_sec = 0.0
        onset = is_fall and not self._prev_fall
        if onset:
            self.event_count += 1
            if self.first_event_sec is None:
                self.first_event_sec = time_sec
        self._prev_fall = is_fall
        return onset

    def update_event(self, is_fall: bool, time_sec: float) -> FallEvent | None:
        if not self.update_signal(is_fall, time_sec):
            return None
        first_event_sec = self.first_event_sec
        if first_event_sec is None:
            first_event_sec = time_sec
        return FallEvent(self.event_count, time_sec, first_event_sec)


def _observation_is_fall(observation: FrameObservation) -> bool:
    return any(label.is_fall for label in observation.labels)


def _observation_fall_probability(observation: FrameObservation) -> float:
    return max((label.confidence for label in observation.labels if label.is_fall), default=1.0)
