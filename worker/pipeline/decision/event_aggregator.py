from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from worker.interfaces.decision import Decider
from worker.pipeline.decision.incident_manager import IncidentManager
from worker.types import BusinessEvent, DecisionInput


@dataclass(frozen=True, slots=True)
class EventAggregator:
    deciders: tuple[Decider, ...]
    incidents: IncidentManager
    monotonic: Callable[[], float] = time.monotonic

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        candidates = sorted(
            (
                event
                for decider in self.deciders
                for event in decider.update(input_value)
            ),
            key=_event_order,
        )
        now_sec = self.monotonic()
        admitted = (
            self.incidents.admit(event, now_sec=now_sec) for event in candidates
        )
        return tuple(event for event in admitted if event is not None)


def _event_order(event: BusinessEvent) -> tuple[str, ...]:
    identity_kind, identity_value = _identity_order(event.identity)
    return (
        event.camera_id,
        event.facility_id,
        event.domain,
        event.event_type,
        f"{event.time_sec:.17g}",
        "" if event.bed_id is None else str(event.bed_id),
        "" if event.person_id is None else str(event.person_id),
        identity_kind,
        identity_value,
    )


def _identity_order(identity: str | int) -> tuple[str, str]:
    return (type(identity).__name__, str(identity))


__all__ = ["EventAggregator"]
