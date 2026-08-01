from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo


class _NightWindowError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NightWindow:
    start: str
    end: str
    tz: str

    def contains(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise _NightWindowError("now must be timezone-aware")
        local_time = now.astimezone(ZoneInfo(self.tz)).time()
        start = _parse_hhmm(self.start)
        end = _parse_hhmm(self.end)
        if start <= end:
            return start <= local_time < end
        return local_time >= start or local_time < end


def _parse_hhmm(value: str) -> time:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise _NightWindowError("night window time must use HH:MM") from exc
    return parsed.time()


__all__ = ["NightWindow"]
