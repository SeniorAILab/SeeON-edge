"""Backend relay payload primitives shared across API and events clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeAlias

AlertEventType: TypeAlias = Literal["fall", "detection-lost", "bed-exit"]
RelayAlertPayload: TypeAlias = dict[str, str | int | float]
RelayHeartbeatPayload: TypeAlias = dict[str, str]


@dataclass(frozen=True, slots=True)
class EventApiPayload:
    camera_id: str
    type: AlertEventType
    detected_at: str
    confidence: float
    config_version: int | None = None
    model_version: str | None = None
    detector_version: str | None = None
    operating_threshold: float | None = None
    clock_source: str | None = None

    def as_dict(self) -> RelayAlertPayload:
        payload: RelayAlertPayload = {
            "camera_id": self.camera_id,
            "type": self.type,
            "detected_at": self.detected_at,
            "confidence": self.confidence,
        }
        if self.config_version is not None:
            payload["config_version"] = self.config_version
        if self.model_version is not None:
            payload["model_version"] = self.model_version
        if self.detector_version is not None:
            payload["detector_version"] = self.detector_version
        if self.operating_threshold is not None:
            payload["operating_threshold"] = self.operating_threshold
        if self.clock_source is not None:
            payload["clock_source"] = self.clock_source
        return payload

    def as_json_bytes(self) -> bytes:
        return json.dumps(self.as_dict(), separators=(",", ":")).encode("utf-8")


__all__ = [
    "AlertEventType",
    "EventApiPayload",
    "RelayAlertPayload",
    "RelayHeartbeatPayload",
]
