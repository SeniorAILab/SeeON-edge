from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class DeliveryDisposition(StrEnum):
    RETRY = "RETRY"
    PERMANENT = "PERMANENT"
    COMPATIBILITY = "COMPATIBILITY"


class DeliveryFailureCode(StrEnum):
    CAMERA_MAPPING_MISSING = "CAMERA_MAPPING_MISSING"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    event_idempotency: Literal[1]
    clip_export: Literal[0, 1]


@dataclass(frozen=True, slots=True)
class DeliveryFailure:
    disposition: DeliveryDisposition
    code: str
    status_code: int | None = None
    retry_after_seconds: float | None = None
    transport_error: str | None = None


@dataclass(frozen=True, slots=True)
class EventReceipt:
    status: Literal["accepted", "accepted_local"]
    edge_event_id: str
    event_id: str


@dataclass(frozen=True, slots=True)
class ClipReceipt:
    clip_id: str
    state: Literal["READY", "UNAVAILABLE", "EXPIRED"]
    state_version: int
    sha256: str | None
    size_bytes: int | None


__all__ = [
    "BackendCapabilities",
    "ClipReceipt",
    "DeliveryDisposition",
    "DeliveryFailure",
    "DeliveryFailureCode",
    "EventReceipt",
]
