from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue


class RelayEvidenceProjectionError(RuntimeError):
    ...


class RelayEvidenceProjectionConflict(RelayEvidenceProjectionError):
    ...


class RelayEvidenceProjectionMissingEvent(RelayEvidenceProjectionError):
    ...


@dataclass(frozen=True, slots=True)
class RelayEvent:
    edge_event_id: str
    event_type: str
    probability: float
    detected_at: str
    camera_id: str
    facility_id: str
    resident_id: str | None
    evidence: dict[str, JsonValue] | None
    audit: dict[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class RelaySnapshot:
    snapshot_id: str
    path: str
    sha256: str
    size_bytes: int
    mime_type: str
    captured_at: str


def _validate_snapshot(snapshot: RelaySnapshot) -> None:
    if snapshot.size_bytes <= 0:
        raise RelayEvidenceProjectionError("snapshot size_bytes must be positive")
    if (
        not snapshot.path
        or snapshot.path.startswith("/")
        or "\\" in snapshot.path
        or snapshot.path in {".", ".."}
        or "/../" in f"/{snapshot.path}/"
    ):
        raise RelayEvidenceProjectionError("snapshot path is not contained")


def _snapshot_timestamp(snapshot_id: str) -> str:
    del snapshot_id
    return "1970-01-01T00:00:00Z"


def _bounded_reason(disposition: str, reason: str) -> str:
    value = f"{disposition}:{reason}"
    return value[:64]


__all__ = [
    "RelayEvent",
    "RelayEvidenceProjectionConflict",
    "RelayEvidenceProjectionError",
    "RelayEvidenceProjectionMissingEvent",
    "RelaySnapshot",
]
