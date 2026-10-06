from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Protocol

from contracts.observation import FrameObservation
from worker.pipeline.output.evidence.evidence_metadata import (
    RUNTIME_MANIFEST_SHA256_KEY,
    validate_runtime_manifest_sha256,
)
from worker.types import BusinessEvent, FramePacket

LOGGER: Final = logging.getLogger(__name__)


class SnapshotRenderer(Protocol):
    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[Any, ...] = (),
    ) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class AlertEvidenceAttacher:
    domain_audit: Mapping[str, Mapping[str, object]]
    snapshot_renderer: SnapshotRenderer | None = None
    debug_snapshots_provider: Callable[[int], tuple[Any, ...]] | None = None
    runtime_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_runtime_manifest_sha256(self.runtime_manifest_sha256)

    def attach_native(self, event: BusinessEvent, snapshot_jpeg: bytes | None) -> BusinessEvent:
        audit = dict(event.audit or {})
        audit.update(self.domain_audit.get(event.domain, {}))
        if self.runtime_manifest_sha256 is not None:
            audit[RUNTIME_MANIFEST_SHA256_KEY] = self.runtime_manifest_sha256
        return replace(
            event,
            audit=audit or None,
            snapshot_jpeg=snapshot_jpeg,
        )

    def attach(
        self,
        event: BusinessEvent,
        packet: FramePacket,
        observation: FrameObservation,
    ) -> BusinessEvent:
        audit = dict(event.audit or {})
        audit.update(self.domain_audit.get(event.domain, {}))
        if self.runtime_manifest_sha256 is not None:
            audit[RUNTIME_MANIFEST_SHA256_KEY] = self.runtime_manifest_sha256
        if not audit:
            return event
        try:
            snapshot_jpeg = None
            if self.snapshot_renderer is not None:
                debug_snapshots = (
                    ()
                    if self.debug_snapshots_provider is None
                    else self.debug_snapshots_provider(packet.frame.index)
                )
                snapshot_jpeg = self.snapshot_renderer.encode_jpeg_bounded(
                    packet, observation, debug_snapshots
                )
            return replace(event, audit=audit, snapshot_jpeg=snapshot_jpeg)
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "failed to attach audit/snapshot metadata to event: camera_id=%s domain=%s",
                event.camera_id,
                event.domain,
                extra={"camera_id": event.camera_id, "domain": event.domain},
            )
            return event


__all__ = ["AlertEvidenceAttacher", "SnapshotRenderer"]
