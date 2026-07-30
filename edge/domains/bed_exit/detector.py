from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from contracts.event import MutableEventPayload
from contracts.observation import BoundingBox
from edge.domains.base import DomainDetector
from edge.domains.bed_exit.schema import (
    BedExitDebugSnapshot,
    BedExitEvent,
    BedExitFrame,
    BedStatus,
)
from edge.perception.domain_input import DomainInput


@dataclass(slots=True)
class _Assignment:
    bed_id: int | None = None
    candidate_bed_id: int | None = None
    candidate_frames: int = 0
    grace_frames: int = 0


@dataclass(frozen=True, slots=True)
class NightWindow:
    start: str
    end: str
    tz: str

    def contains(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        local_now = now.astimezone(ZoneInfo(self.tz)).time()
        start = _parse_hhmm(self.start)
        end = _parse_hhmm(self.end)
        if start <= end:
            return start <= local_now < end
        return local_now >= start or local_now < end


class BedExitMonitor(DomainDetector):
    """Sticky per-person bed assignment and own-bed exit detector."""

    enabled = True

    def __init__(
        self,
        *,
        min_containment: float = 0.35,
        hold_frames: int = 2,
        grace_frames: int = 3,
        night_window: NightWindow | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if min_containment <= 0.0 or min_containment > 1.0:
            raise ValueError("min_containment must be in (0, 1]")
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if grace_frames < 0:
            raise ValueError("grace_frames must be >= 0")
        self._min_containment = min_containment
        self._hold_frames = hold_frames
        self._grace_frames = grace_frames

        self._assignments: dict[int, _Assignment] = {}
        self._night_window = night_window
        self._clock = (
            clock
            if clock is not None
            else (
                lambda: (
                    datetime.now(ZoneInfo(self._night_window.tz))
                    if self._night_window is not None
                    else None
                )
            )
        )
        self.last_debug_snapshot: BedExitDebugSnapshot | None = None
    def update_night_window(self, night_window: NightWindow | None) -> None:
        self._night_window = night_window

    def update(self, domain_input: DomainInput) -> tuple[MutableEventPayload, ...]:
        observation = domain_input.observation
        if not observation.bed_boxes:
            self.last_debug_snapshot = BedExitDebugSnapshot(
                frame_index=domain_input.frame_index,
                person_boxes=observation.boxes,
                bed_boxes=(),
                statuses=(),
                events=(),
                bed_region=domain_input.bed_region,
            )
            return ()
        frame = self.update_boxes(
            bed_boxes=observation.bed_boxes,
            person_boxes=observation.boxes,
            person_ids=observation.track_ids,
        )
        self.last_debug_snapshot = BedExitDebugSnapshot(
            frame_index=domain_input.frame_index,
            person_boxes=observation.boxes,
            bed_boxes=observation.bed_boxes,
            statuses=frame.statuses,
            events=frame.events,
            bed_region=domain_input.bed_region,
        )
        if self._night_window is not None:
            assert self._clock is not None
            if not self._night_window.contains(self._clock()):
                return ()
        return tuple(_event_dict(event, domain_input.time_sec) for event in frame.events)

    def update_boxes(
        self,
        *,
        bed_boxes: tuple[BoundingBox, ...],
        person_boxes: tuple[BoundingBox, ...],
        person_ids: tuple[int | None, ...] = (),
    ) -> BedExitFrame:
        if not person_ids:
            person_ids = tuple(range(len(person_boxes)))
        live_ids = {person_id for person_id in person_ids if person_id is not None}
        for stale_id in set(self._assignments) - live_ids:
            self._assignments.pop(stale_id, None)

        events: list[BedExitEvent] = []
        occupied: dict[int, int] = {}
        exit_beds: set[int] = set()

        for person_id, person_box in zip(person_ids, person_boxes, strict=True):
            if person_id is None:
                continue
            assignment = self._assignments.setdefault(person_id, _Assignment())
            containments = tuple(_containment_ratio(person_box, bed) for bed in bed_boxes)
            best_bed_id = _best_bed_id(containments, self._min_containment)

            if assignment.bed_id is None:
                self._update_candidate(assignment, best_bed_id)
                if assignment.candidate_frames >= self._hold_frames:
                    assignment.bed_id = assignment.candidate_bed_id
                    assignment.grace_frames = 0
                if assignment.bed_id is not None:
                    occupied[assignment.bed_id] = person_id
                continue

            own_ratio = (
                containments[assignment.bed_id] if assignment.bed_id < len(containments) else 0.0
            )
            in_own_bed = own_ratio >= self._min_containment
            in_other_bed = any(
                bed_id != assignment.bed_id and ratio >= self._min_containment
                for bed_id, ratio in enumerate(containments)
            )

            if in_own_bed:
                assignment.grace_frames = 0
                occupied[assignment.bed_id] = person_id
                continue

            if in_other_bed:
                assignment.grace_frames = 0
                continue

            assignment.grace_frames += 1
            if assignment.grace_frames > self._grace_frames:
                exited_bed_id = assignment.bed_id
                events.append(BedExitEvent(person_id=person_id, bed_id=exited_bed_id))
                exit_beds.add(exited_bed_id)
                assignment.bed_id = None
                assignment.candidate_bed_id = None
                assignment.candidate_frames = 0
                assignment.grace_frames = 0

        statuses = tuple(
            BedStatus(
                bed_id=bed_id,
                box=bed_box,
                occupancy="exit"
                if bed_id in exit_beds
                else "occupied"
                if bed_id in occupied
                else "empty",
                person_id=occupied.get(bed_id),
            )
            for bed_id, bed_box in enumerate(bed_boxes)
        )
        return BedExitFrame(statuses=statuses, events=tuple(events))

    def _update_candidate(self, assignment: _Assignment, bed_id: int | None) -> None:
        if bed_id is None:
            assignment.candidate_bed_id = None
            assignment.candidate_frames = 0
            return
        if assignment.candidate_bed_id == bed_id:
            assignment.candidate_frames += 1
        else:
            assignment.candidate_bed_id = bed_id
            assignment.candidate_frames = 1


def _parse_hhmm(value: str) -> time:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError("night window time must use HH:MM") from exc
    return parsed.time()


def _best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = [
        (ratio, bed_id) for bed_id, ratio in enumerate(containments) if ratio >= min_containment
    ]
    if not candidates:
        return None
    ratio, bed_id = max(candidates, key=lambda item: (item[0], -item[1]))
    return bed_id if ratio >= min_containment else None


def _containment_ratio(person: BoundingBox, bed: BoundingBox) -> float:
    left = max(person.x1, bed.x1)
    top = max(person.y1, bed.y1)
    right = min(person.x2, bed.x2)
    bottom = min(person.y2, bed.y2)
    width = max(0, right - left)
    height = max(0, bottom - top)
    intersection = width * height
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if intersection == 0 or person_area <= 0:
        return 0.0
    return intersection / person_area


def _event_dict(event: BedExitEvent, time_sec: float | None) -> MutableEventPayload:
    return {
        "domain": "bed_exit",
        "event_type": "bed-exit",
        "identity": f"{event.person_id}:{event.bed_id}",
        "person_id": event.person_id,
        "bed_id": event.bed_id,
        "probability": 1.0,
        "time_sec": 0.0 if time_sec is None else time_sec,
    }
