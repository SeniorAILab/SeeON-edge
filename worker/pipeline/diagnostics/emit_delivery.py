from __future__ import annotations

from pathlib import Path
from typing import Final

from shared.events.execution_records import PROCESS_SCOPE, WireRecord
from worker.pipeline.diagnostics.record_builder import (
    PRODUCER_BACKEND,
    PRODUCER_EVENT,
    WALL,
    make_record,
    wall_or,
)

DELIVERY_ATTEMPT_OUTCOMES: Final = (
    "retry-transient",
    "retry-counted",
    "refused-retained",
    "refused-retention-full",
    "exhausted-retained",
    "exhausted-retention-full",
    "ack-removal-deferred",
)


def event_delivery_record(
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    frame_seq: int,
    source_pts_ns: int | None,
    edge_event_id: str,
    event_type: str,
    domain: str,
    admitted: bool,
    reason: str | None = None,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    return make_record(
        record_kind="event.delivery",
        camera_id=camera_id,
        worker_boot_id=worker_boot_id,
        source_generation=source_generation,
        stream_epoch=stream_epoch,
        producer=PRODUCER_EVENT,
        observed_at_ns=wall_or(observed_at_ns),
        time_quality=WALL,
        causal_unit_id=edge_event_id,
        outcome="admitted" if admitted else "refused",
        payload={
            "edge_event_id": edge_event_id,
            "event_type": event_type,
            "domain": domain,
            "queue": "delivery",
            "reason": reason,
        },
        frame_seq=frame_seq,
        source_pts_ns=source_pts_ns,
    )


def delivery_attempt_record(
    *,
    camera_id: str,
    observing_boot_id: str,
    edge_event_id: str,
    outcome: str,
    attempt: int,
    max_attempts: int,
    failure_class: str | None,
    status_code: int | None,
    retained: bool | None,
    dead_letter_dir: str | None = None,
    queue_kind: str = "EVENT",
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    if outcome not in DELIVERY_ATTEMPT_OUTCOMES:
        return None
    dir_name = None if dead_letter_dir is None else Path(dead_letter_dir).name
    return make_record(
        record_kind="event.delivery",
        camera_id=camera_id,
        worker_boot_id=observing_boot_id,
        source_generation=PROCESS_SCOPE,
        stream_epoch=PROCESS_SCOPE,
        producer=PRODUCER_EVENT,
        observed_at_ns=wall_or(observed_at_ns),
        time_quality=WALL,
        causal_unit_id=edge_event_id,
        outcome=outcome,
        payload={
            "edge_event_id": edge_event_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "failure_class": failure_class,
            "status_code": status_code,
            "retained": retained,
            "dead_letter_dir": dir_name,
            "queue_kind": queue_kind,
        },
    )


def backend_acceptance_record(
    *,
    camera_id: str,
    observing_boot_id: str,
    edge_event_id: str,
    status: str,
    hub_event_id: str,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    if status == "accepted_local":
        outcome = "accepted_local"
    elif status == "accepted":
        outcome = "hub-accepted"
    else:
        outcome = status
    return make_record(
        record_kind="backend.acceptance",
        camera_id=camera_id,
        worker_boot_id=observing_boot_id,
        source_generation=PROCESS_SCOPE,
        stream_epoch=PROCESS_SCOPE,
        producer=PRODUCER_BACKEND,
        observed_at_ns=wall_or(observed_at_ns),
        time_quality=WALL,
        causal_unit_id=edge_event_id,
        outcome=outcome,
        payload={
            "edge_event_id": edge_event_id,
            "status": status,
            "hub_event_id": hub_event_id,
            "accepted_local": status == "accepted_local",
            "hub_accepted": status == "accepted" and bool(hub_event_id),
            "observing_boot_id": observing_boot_id,
            "origin_boot_id": None,
            "origin_source_generation": None,
            "origin_stream_epoch": None,
        },
    )


__all__ = [
    "DELIVERY_ATTEMPT_OUTCOMES",
    "backend_acceptance_record",
    "delivery_attempt_record",
    "event_delivery_record",
]
