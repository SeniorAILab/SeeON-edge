from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, TypeAlias, override
from uuid import uuid4

from shared.boundary import Boundary, LogThrottle, isolate
from worker.pipeline.decision.event_identity import EventIdentityStore
from worker.types import BusinessEvent

_LOGGER: Final = logging.getLogger(__name__)

CooldownKey: TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class IncidentAuditSnapshot:
    edge_event_id: str
    source_identity: str | int
    cooldown_key: CooldownKey
    domain: str
    event_type: str
    camera_id: str
    facility_id: str
    time_sec: float
    probability: float
    person_id: int | None
    bed_id: int | None


@dataclass(slots=True)
class IncidentConfigurationError(ValueError):
    cooldown_sec: float

    @override
    def __str__(self) -> str:
        return f"cooldown_sec must be non-negative, received {self.cooldown_sec}"


@dataclass(slots=True)
class IncidentManager:
    cooldown_sec: float = 30.0
    identity_path: Path | None = None
    _last_seen: dict[CooldownKey, float] = field(default_factory=dict, init=False)
    _admitted_keys: dict[str, CooldownKey] = field(default_factory=dict, init=False)
    _identities: EventIdentityStore = field(init=False, repr=False)
    identity_journal_failures: int = field(default=0, init=False)
    _journal_throttle: LogThrottle = field(default_factory=LogThrottle, init=False, repr=False)
    cooldown_suppressed_total: int = field(default=0, init=False)
    last_audit_snapshot: IncidentAuditSnapshot | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.cooldown_sec < 0.0:
            raise IncidentConfigurationError(self.cooldown_sec)
        with isolate(
            Boundary.OPTIONAL_FEATURE, stage="event_identity_journal_open", level=logging.ERROR
        ) as opened:
            self._identities = EventIdentityStore(self.identity_path)
        if opened.failed:
            self._identities = EventIdentityStore(None)
            self.identity_journal_failures += 1

    def admit(
        self,
        event: BusinessEvent,
        *,
        now_sec: float | None = None,
    ) -> BusinessEvent | None:
        event_time = event.time_sec if now_sec is None else now_sec
        key = self.idempotency_key(event, event_time)
        last_seen = self._last_seen.get(key)
        if last_seen is not None and event_time - last_seen < self.cooldown_sec:
            self.cooldown_suppressed_total += 1
            return None

        source_identity = event.identity
        edge_event_id = ""
        with isolate(
            Boundary.OPTIONAL_FEATURE,
            stage="event_identity_journal_resolve",
            throttle=self._journal_throttle,
            level=logging.ERROR,
            camera_id=event.camera_id,
            event_type=event.event_type,
        ) as resolved:
            edge_event_id = self._identities.resolve(_source_key(event))
        if resolved.failed:
            edge_event_id = str(uuid4())
            self.identity_journal_failures += 1
        admitted = replace(event, identity=edge_event_id)
        self._last_seen[key] = event_time
        self._admitted_keys[edge_event_id] = key
        self.last_audit_snapshot = IncidentAuditSnapshot(
            edge_event_id=edge_event_id,
            source_identity=source_identity,
            cooldown_key=key,
            domain=event.domain,
            event_type=event.event_type,
            camera_id=event.camera_id,
            facility_id=event.facility_id,
            time_sec=event.time_sec,
            probability=event.probability,
            person_id=event.person_id,
            bed_id=event.bed_id,
        )
        return admitted

    def register(
        self,
        event: BusinessEvent,
        *,
        now_sec: float | None = None,
    ) -> BusinessEvent | None:
        return self.admit(event, now_sec=now_sec)

    def release(self, event: BusinessEvent, *, now_sec: float | None = None) -> None:
        del now_sec
        key = self._admitted_keys.pop(str(event.identity), None)
        if key is not None:
            self._last_seen.pop(key, None)

    def idempotency_key(
        self,
        event: BusinessEvent,
        event_time: float | None = None,
    ) -> CooldownKey:
        del event_time
        return (event.camera_id, event.domain, event.event_type, event.identity)

    def reset(self) -> None:
        self._last_seen.clear()
        self.last_audit_snapshot = None


def _source_key(event: BusinessEvent) -> str:
    return json.dumps(
        [
            event.facility_id,
            event.camera_id,
            event.domain,
            event.event_type,
            event.identity,
            event.time_sec,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


__all__ = [
    "CooldownKey",
    "IncidentAuditSnapshot",
    "IncidentConfigurationError",
    "IncidentManager",
]
