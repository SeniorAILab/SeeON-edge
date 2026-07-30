"""L0 event vocabulary aligned to the frontend domain SSoT.

ML emits typed signed events. Backend owns severity/channel/policy/final-dedup;
ML runtime incident management owns only idempotency and cooldown.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeAlias

EventScalar: TypeAlias = str | int | float | bool | None
EventEvidence: TypeAlias = Mapping[str, EventScalar]
EventPayload: TypeAlias = Mapping[str, EventScalar | EventEvidence]
MutableEventPayload: TypeAlias = dict[str, EventScalar | dict[str, EventScalar]]


class Level(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


Severity = Level


class DetectionEventType(StrEnum):
    STABLE = "STABLE"
    MOVEMENT_INCREASE = "MOVEMENT_INCREASE"
    REPEATED_STANDING_ATTEMPT = "REPEATED_STANDING_ATTEMPT"
    FALL_RISK = "FALL_RISK"
    SOLO_MOVEMENT = "SOLO_MOVEMENT"
    PROLONGED_INACTIVITY = "PROLONGED_INACTIVITY"
    WANDERING = "WANDERING"
    BED_EXIT = "BED_EXIT"
    OTHER = "OTHER"


_DEFAULT_EVENT_TYPE_REGISTRY: Final[dict[str, DetectionEventType]] = {
    "stable": DetectionEventType.STABLE,
    "movement-increase": DetectionEventType.MOVEMENT_INCREASE,
    "repeated-standing-attempt": DetectionEventType.REPEATED_STANDING_ATTEMPT,
    "fall": DetectionEventType.FALL_RISK,
    "fall-risk": DetectionEventType.FALL_RISK,
    "solo-movement": DetectionEventType.SOLO_MOVEMENT,
    "prolonged-inactivity": DetectionEventType.PROLONGED_INACTIVITY,
    "wandering": DetectionEventType.WANDERING,
    "bed-exit": DetectionEventType.BED_EXIT,
    "detection-lost": DetectionEventType.OTHER,
    "other": DetectionEventType.OTHER,
}
_EVENT_TYPE_REGISTRY: dict[str, DetectionEventType] = dict(_DEFAULT_EVENT_TYPE_REGISTRY)
EVENT_TYPE_REGISTRY = MappingProxyType(_EVENT_TYPE_REGISTRY)


def register_event_type(event_type: str, front_event_type: DetectionEventType | str) -> None:
    normalized = event_type.strip()
    if normalized == "":
        raise ValueError("event_type must be non-empty")
    _EVENT_TYPE_REGISTRY[normalized] = DetectionEventType(front_event_type)


def front_event_type(event_type: str) -> DetectionEventType:
    return _EVENT_TYPE_REGISTRY.get(event_type, DetectionEventType.OTHER)


__all__ = [
    "DetectionEventType",
    "EventEvidence",
    "EventPayload",
    "EventScalar",
    "EVENT_TYPE_REGISTRY",
    "Level",
    "MutableEventPayload",
    "Severity",
    "front_event_type",
    "register_event_type",
]
