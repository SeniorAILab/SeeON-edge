from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from contracts.observation import BedRegionDebugSnapshot, BoundingBox

BedOccupancy = Literal["empty", "occupied", "exit"]


@dataclass(frozen=True, slots=True)
class BedStatus:
    bed_id: int
    box: BoundingBox
    occupancy: BedOccupancy
    person_id: int | None = None


@dataclass(frozen=True, slots=True)
class BedExitEvent:
    person_id: int
    bed_id: int


@dataclass(frozen=True, slots=True)
class BedExitFrame:
    statuses: tuple[BedStatus, ...]
    events: tuple[BedExitEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BedExitDebugSnapshot:
    frame_index: int | None
    person_boxes: tuple[BoundingBox, ...]
    bed_boxes: tuple[BoundingBox, ...]
    statuses: tuple[BedStatus, ...]
    events: tuple[BedExitEvent, ...] = field(default_factory=tuple)
    bed_region: BedRegionDebugSnapshot | None = None


@dataclass(frozen=True, slots=True)
class DomainDebugSnapshot:
    domain: str
    bed_exit: BedExitDebugSnapshot | None = None
