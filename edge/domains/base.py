from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from contracts.event import MutableEventPayload
from contracts.observation import BedRegionDebugSnapshot, FrameObservation


class DomainInputProtocol(Protocol):
    """Structural view supplied by the shared perception pipeline."""

    observation: FrameObservation
    frame_width: int
    frame_height: int
    live_track_ids: tuple[int, ...]
    time_sec: float | None
    frame_index: int
    bed_region: BedRegionDebugSnapshot


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Immutable model audit context supplied by worker composition."""

    model_version: str | None
    operating_threshold: float | None


class DomainDetector(ABC):
    """Common interface for domain-level interpretation of prepared inputs."""

    enabled: bool
    audit_context: AuditContext | None = None

    @abstractmethod
    def update(self, domain_input: DomainInputProtocol) -> tuple[MutableEventPayload, ...]:
        """Interpret one prepared frame and return domain events."""

    def audit_metadata(self) -> Mapping[str, object]:
        """Return detector-owned audit metadata for emitted events."""
        return {}
