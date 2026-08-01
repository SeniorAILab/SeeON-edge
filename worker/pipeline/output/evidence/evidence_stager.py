"""Durably stage canonical relay alerts before recorder side effects."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contracts.event import EventPayload
from contracts.worker_config import CONFIG_VERSION_KEY
from worker.pipeline.output.evidence.evidence_outbox import (
    ClipId,
    EdgeEventId,
    EvidenceOutbox,
    StagedEvent,
)


@dataclass(frozen=True, slots=True)
class DurableEvidenceStager:
    database_path: Path
    camera_id: str
    facility_id: str
    resident_id: str | None
    config_version: int
    clock: Callable[[], float]

    def stage(self, event: EventPayload) -> None:
        edge_event_id = EdgeEventId(_required_text(event, "edge_event_id"))
        detected_at = _required_text(event, "detected_at")
        payload = self._canonical_payload(event, edge_event_id, detected_at)
        with EvidenceOutbox.open(self.database_path) as outbox:
            outbox.stage(
                StagedEvent(
                    edge_event_id=edge_event_id,
                    detected_at=detected_at,
                    payload_json=json.dumps(
                        payload,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    queued_at=self.clock(),
                )
            )

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        event_id = EdgeEventId(edge_event_id)
        with EvidenceOutbox.open(self.database_path) as outbox:
            if clip_id is None:
                outbox.mark_ready(event_id)
                return
            outbox.bind_clip(event_id, ClipId(clip_id))

    def _canonical_payload(
        self,
        event: EventPayload,
        edge_event_id: EdgeEventId,
        detected_at: str,
    ) -> dict[str, object]:
        event_type = _required_text(event, "event_type")
        evidence = dict(event)
        event_audit = evidence.pop("audit", None)
        snapshot_jpeg = evidence.pop("snapshot_jpeg", None)
        snapshot = evidence.pop("snapshot", None)
        payload: dict[str, object] = {
            "edge_event_id": edge_event_id,
            "event_type": "bed-exit" if event_type == "bed-exit" else "fall",
            "probability": _event_probability(event),
            "detected_at": detected_at,
            "camera_id": self.camera_id,
            "facility_id": self.facility_id,
            "evidence": evidence,
        }
        if self.resident_id is not None:
            payload["resident_id"] = self.resident_id
        if isinstance(event_audit, Mapping):
            payload["audit"] = {**event_audit, CONFIG_VERSION_KEY: self.config_version}
        if isinstance(snapshot_jpeg, bytes):
            payload["snapshot_jpeg_base64"] = base64.b64encode(snapshot_jpeg).decode(
                "ascii"
            )
        if isinstance(snapshot, Mapping):
            payload["snapshot"] = dict(snapshot)
        return payload


def _required_text(event: EventPayload, key: str) -> str:
    value = event.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"event {key} must be set")
    return text


def _event_probability(event: EventPayload) -> float:
    value = event.get("probability", event.get("confidence", 1.0))
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    return 1.0


__all__ = ["DurableEvidenceStager"]
