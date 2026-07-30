from __future__ import annotations

from dataclasses import dataclass, field


def _event_value(event: object, name: str, default: object = None) -> object:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


@dataclass(slots=True)
class IncidentManager:
    """Idempotency and cooldown gate for emitted incident events.

    This class deliberately does not assign severity, apply policy, route events,
    or fan out notifications. It only answers whether an event is new enough to
    emit.
    """

    cooldown_sec: float = 30.0
    bucket_sec: float | None = None
    _last_seen: dict[tuple[object, ...], float] = field(default_factory=dict, init=False)

    def admit(self, event: object, *, now_sec: float | None = None) -> bool:
        event_time = _coerce_time(now_sec, _event_value(event, "time_sec", 0.0))
        key = self.idempotency_key(event, event_time)
        if key is None:
            return True
        last_seen = self._last_seen.get(key)
        if last_seen is not None and event_time - last_seen < self.cooldown_sec:
            return False
        self._last_seen[key] = event_time
        return True

    def register(self, event: object, *, now_sec: float | None = None) -> bool:
        return self.admit(event, now_sec=now_sec)

    def idempotency_key(
        self, event: object, event_time: float
    ) -> tuple[object, ...] | None:
        explicit_key = _event_value(event, "idempotency_key")
        if explicit_key is not None:
            return (explicit_key,)

        del event_time
        camera_id = _event_value(event, "camera_id")
        domain = _event_value(event, "domain")
        event_type = _event_value(event, "event_type", _event_value(event, "type"))
        if domain == "fall":
            return (camera_id, domain, event_type)
        if domain == "bed_exit":
            bed_id = _event_value(event, "bed_id")
            if bed_id is not None:
                return (camera_id, domain, event_type, bed_id)

        identity = _event_value(event, "identity")
        if identity is None:
            identity = _event_value(event, "event_id")
        if identity is None:
            return None
        return (
            camera_id,
            domain,
            event_type,
            identity,
        )


    def reset(self) -> None:
        self._last_seen.clear()


def _coerce_time(now_sec: float | None, event_time: object) -> float:
    if now_sec is not None:
        return float(now_sec)
    if isinstance(event_time, int | float):
        return float(event_time)
    return 0.0
