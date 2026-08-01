from __future__ import annotations

from worker.domains.bed_exit.schema import BedExitEvent


class BedExitLatch:
    """Return only bed-exit transitions absent from the previous frame."""

    def __init__(self) -> None:
        self.event_count: int = 0
        self.first_event_sec: float | None = None
        self._active_exits: set[tuple[int, int]] = set()

    def update(
        self,
        events: tuple[BedExitEvent, ...],
        time_sec: float,
    ) -> tuple[BedExitEvent, ...]:
        event_keys = {(event.person_id, event.bed_id) for event in events}
        onset_events = tuple(
            event for event in events if (event.person_id, event.bed_id) not in self._active_exits
        )
        if onset_events:
            self.event_count += len(onset_events)
            if self.first_event_sec is None:
                self.first_event_sec = time_sec
        self._active_exits = event_keys
        return onset_events


__all__ = ["BedExitLatch"]
