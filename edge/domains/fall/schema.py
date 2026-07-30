from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FallEvent:
    event_count: int
    onset_sec: float
    first_event_sec: float
