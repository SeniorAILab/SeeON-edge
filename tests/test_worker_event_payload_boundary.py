from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import get_args, get_type_hints

from contracts.event import EventPayload, EventScalar, MutableEventPayload
from worker.pipeline.output.evidence.event_payload import (
    MutableWorkerEventPayload,
    WorkerEventPayload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_EVENT_SHA256 = "eeb7e038ef926d4f68f31f2c0571c4d823fb1da38458a569377719a173e89158"


def test_contract_event_module_matches_pinned_digest() -> None:
    content = (REPO_ROOT / "contracts" / "event.py").read_bytes()

    assert hashlib.sha256(content).hexdigest() == CANONICAL_EVENT_SHA256


def test_canonical_event_payload_stays_a_bytes_free_mapping_alias() -> None:
    key_type, value_type = get_args(EventPayload)
    assert key_type is str
    assert bytes not in _flatten_union(value_type)

    mutable_value_type = get_args(MutableEventPayload)[1]
    assert bytes not in _flatten_union(mutable_value_type)
    assert bytes not in _flatten_union(EventScalar)


def _flatten_union(type_arg: object) -> tuple[object, ...]:
    args = get_args(type_arg)
    if not args:
        return (type_arg,)
    flattened: list[object] = []
    for arg in args:
        flattened.extend(_flatten_union(arg))
    return tuple(flattened)


def test_worker_local_staging_payload_requires_strict_bytes_snapshot_field() -> None:
    hints = get_type_hints(WorkerEventPayload, include_extras=True)
    snapshot_hint = hints["snapshot_jpeg"]

    assert get_args(snapshot_hint) == (bytes,)

    payload: WorkerEventPayload = {
        "edge_event_id": "event-1",
        "snapshot_jpeg": b"jpeg-bytes",
    }
    assert isinstance(payload["snapshot_jpeg"], bytes)

    mutable: MutableWorkerEventPayload = {}
    mutable["snapshot_jpeg"] = b"jpeg-bytes"
    assert isinstance(mutable["snapshot_jpeg"], bytes)

    assert isinstance(payload, Mapping)


def test_worker_local_staging_payload_does_not_leak_across_shared_boundary() -> None:
    client_source = (REPO_ROOT / "shared" / "events" / "edge_ingest_client.py").read_text(
        encoding="utf-8"
    )

    assert "worker.pipeline.output.evidence.event_payload" not in client_source
    assert "WorkerEventPayload" not in client_source
    assert "from contracts.event import EventPayload" in client_source


def test_worker_local_staging_type_is_scoped_under_worker() -> None:
    module_path = Path("worker/pipeline/output/evidence/event_payload.py")

    assert (REPO_ROOT / module_path).is_file()
    assert WorkerEventPayload.__module__ == "worker.pipeline.output.evidence.event_payload"
