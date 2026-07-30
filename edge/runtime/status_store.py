from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from time import time


class CameraStatus(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True, slots=True)
class CameraStatusRecord:
    camera_id: str
    facility_id: str
    status: CameraStatus
    updated_at: float
    error_category: str | None = None


@dataclass(frozen=True, slots=True)
class OpsEvent:
    event_type: str
    camera_id: str
    facility_id: str
    category: str
    timestamp: float
    detail: str | None = None


@dataclass(slots=True)
class StatusStore:
    _statuses: dict[str, CameraStatusRecord] = field(default_factory=dict)
    _ops_events: list[OpsEvent] = field(default_factory=list)

    def set_status(
        self,
        camera_id: str,
        facility_id: str,
        status: CameraStatus | str,
        *,
        error_category: str | None = None,
        timestamp: float | None = None,
    ) -> CameraStatusRecord:
        record = CameraStatusRecord(
            camera_id=camera_id,
            facility_id=facility_id,
            status=CameraStatus(status),
            updated_at=_timestamp(timestamp),
            error_category=error_category,
        )
        self._statuses[camera_id] = record
        return record

    def get_status(self, camera_id: str) -> CameraStatusRecord | None:
        return self._statuses.get(camera_id)

    def record_ops_event(
        self,
        event_type: str,
        camera_id: str,
        facility_id: str,
        category: str,
        *,
        timestamp: float | None = None,
        detail: str | None = None,
    ) -> OpsEvent:
        event = OpsEvent(
            event_type=event_type,
            camera_id=camera_id,
            facility_id=facility_id,
            category=category,
            timestamp=_timestamp(timestamp),
            detail=detail,
        )
        self._ops_events.append(event)
        return event

    def record_camera_reconnecting(
        self,
        camera_id: str,
        facility_id: str,
        category: str,
        *,
        detail: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.set_status(
            camera_id,
            facility_id,
            CameraStatus.DEGRADED,
            error_category=category,
            timestamp=timestamp,
        )
        self.record_ops_event(
            "camera.offline",
            camera_id,
            facility_id,
            category,
            timestamp=timestamp,
            detail=detail,
        )

    def record_camera_recovered(
        self,
        camera_id: str,
        facility_id: str,
        *,
        detail: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.set_status(
            camera_id,
            facility_id,
            CameraStatus.READY,
            timestamp=timestamp,
        )
        self.record_ops_event(
            "camera.recovered",
            camera_id,
            facility_id,
            "rtsp_recovered",
            timestamp=timestamp,
            detail=detail,
        )

    def ops_events(self) -> tuple[OpsEvent, ...]:
        return tuple(self._ops_events)

    def snapshot(self) -> dict[str, object]:
        return {
            "cameras": {
                camera_id: _record_dict(record)
                for camera_id, record in sorted(self._statuses.items())
            },
            "ops_events": [_record_dict(event) for event in self._ops_events],
        }


def _timestamp(timestamp: float | None) -> float:
    return time() if timestamp is None else float(timestamp)


def _record_dict(record: CameraStatusRecord | OpsEvent) -> dict[str, object]:
    data = asdict(record)
    if isinstance(record, CameraStatusRecord):
        data["status"] = record.status.value
    return data
