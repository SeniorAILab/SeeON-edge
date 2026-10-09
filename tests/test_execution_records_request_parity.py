from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.features.diagnostics.schemas import ExecutionRecordBatchRequest
from shared.events.execution_records import (
    RECORD_KINDS,
    TIME_QUALITIES,
    ExecutionRecordContractError,
    WireBatch,
    WireGap,
    WireProvenance,
    WireRecord,
)
from worker.pipeline.diagnostics import emit_delivery, emit_policy
from worker.pipeline.diagnostics.lanes import (
    EXPORT_FAILED_CAUSE,
    LANE_OVERFLOW_CAUSE,
    RECORD_INVALID_CAUSE,
)
from worker.types.trace import DecisionTraceReason, DecisionTraceSnapshot, DecisionTraceState

Body = dict[str, Any]
Edit = Callable[[Body], None]

_CAMERA = "camera-1"
_BOOT = "boot-1"
_LONG = "x" * 128
_T = 1_760_000_000_000_000_000
_GAP_CAUSES = (LANE_OVERFLOW_CAUSE, EXPORT_FAILED_CAUSE, RECORD_INVALID_CAUSE)


def _provenance(**overrides: str) -> WireProvenance:
    fields = {name: f"{name}-1" for name in WireProvenance.__slots__}
    return WireProvenance(**(fields | overrides))


def _record(sequence: int, **overrides: Any) -> WireRecord:
    kind = overrides.pop("record_kind", "sdk.frame")
    scope = 0 if kind == "backend.acceptance" else 1
    fields: dict[str, Any] = {
        "record_kind": kind,
        "camera_id": _CAMERA,
        "worker_boot_id": _BOOT,
        "source_generation": scope,
        "stream_epoch": scope,
        "producer": "sdk",
        "producer_sequence": sequence,
        "observed_at_ns": _T + sequence,
        "time_quality": "pts",
        "causal_unit_id": f"{_CAMERA}:{_BOOT}:1:frame:{sequence}",
        "outcome": "accepted",
        "payload": {"seq": sequence},
        "frame_seq": sequence,
        "source_pts_ns": 33_366_666 * sequence,
    }
    return WireRecord(**(fields | overrides))


def _gap(**overrides: Any) -> WireGap:
    fields: dict[str, Any] = {
        "producer": "sdk",
        "from_sequence": 3,
        "to_sequence": 5,
        "from_ns": _T,
        "to_ns": _T + 2,
        "record_count": 3,
        "cause": LANE_OVERFLOW_CAUSE,
        "source_generation": 1,
        "stream_epoch": 1,
    }
    return WireGap(**(fields | overrides))


def _wire(
    records: tuple[WireRecord, ...] = (),
    gaps: tuple[WireGap, ...] = (),
    camera_id: str = _CAMERA,
    worker_boot_id: str = _BOOT,
    provenance: WireProvenance | None = None,
) -> Body:
    batch = WireBatch(
        camera_id=camera_id,
        worker_boot_id=worker_boot_id,
        provenance=provenance or _provenance(),
        records=records,
        gaps=gaps,
    )
    loaded: Body = json.loads(batch.encode())
    return loaded


class _Probability:
    fall_transition = 0.25
    background = 0.5
    fallen = 0.25
    model_evidence = None


def _worker_records(pts: int) -> tuple[WireRecord, ...]:
    scope = {
        "camera_id": _CAMERA,
        "worker_boot_id": _BOOT,
        "source_generation": 1,
        "stream_epoch": 2,
    }
    unknown = str(DecisionTraceState.NOT_EVALUATED)
    built = [
        emit_policy.model_score_record(
            **scope,
            frame_seq=1,
            source_pts_ns=pts,
            track_id=7,
            generation=None,
            probability=_Probability(),
            observed_at_ns=_T,
        ),
        *[
            emit_policy.policy_decision_record(
                DecisionTraceSnapshot(
                    reason=str(reason),
                    previous_state=unknown,
                    current_state=unknown,
                    triggered=False,
                    track_id=None,
                    bed_id=None,
                ),
                **scope,
                frame_seq=2,
                source_pts_ns=pts,
                generation=3,
                module_qualified_id="fall",
                authority_role="primary",
                observed_at_ns=_T + 1,
            )
            for reason in DecisionTraceReason
        ],
        emit_policy.policy_coast_record(
            **scope,
            frame_seq=3,
            source_pts_ns=pts,
            module_qualified_id="fall",
            observed_at_ns=_T + 2,
        ),
        emit_delivery.event_delivery_record(
            **scope,
            frame_seq=4,
            source_pts_ns=pts,
            edge_event_id="evt-1",
            event_type="fall",
            domain="fall",
            admitted=False,
            reason="cooldown",
            observed_at_ns=_T + 3,
        ),
        emit_delivery.delivery_attempt_record(
            camera_id=_CAMERA,
            observing_boot_id=_BOOT,
            edge_event_id="evt-1",
            outcome="exhausted-retained",
            attempt=3,
            max_attempts=3,
            failure_class=None,
            status_code=None,
            retained=None,
            observed_at_ns=_T + 4,
        ),
        emit_delivery.backend_acceptance_record(
            camera_id=_CAMERA,
            observing_boot_id=_BOOT,
            edge_event_id="evt-1",
            status="accepted",
            hub_event_id="hub-1",
            observed_at_ns=_T + 5,
        ),
    ]
    records = [record for record in built if record is not None]
    assert len(records) == len(built)
    return tuple(records)


def _edited(body: Body, edit: Edit) -> Body:
    edit(body)
    return body


def _accepted_bodies() -> list[tuple[str, Body]]:
    first = _record(0)
    long_ids = {"camera_id": _LONG, "worker_boot_id": _LONG}
    return [
        *[
            (f"worker-emitters-pts-{pts}", _wire(_worker_records(pts)))
            for pts in (0, 2**64 - 1, -1)
        ],
        (
            "every-kind",
            _wire(
                tuple(_record(i, record_kind=kind) for i, kind in enumerate(sorted(RECORD_KINDS)))
            ),
        ),
        (
            "every-time-quality",
            _wire(tuple(_record(i, time_quality=q) for i, q in enumerate(sorted(TIME_QUALITIES)))),
        ),
        ("null-optionals", _wire((_record(0, frame_seq=None, source_pts_ns=None),))),
        *[
            (f"pts-{pts}", _wire((_record(0, source_pts_ns=pts),)))
            for pts in (-1, -(2**63), 0, 2**63 - 1, 2**64 - 1)
        ],
        *[
            (f"reason-{reason}", _wire((_record(0, reason=str(reason)),)))
            for reason in DecisionTraceReason
        ],
        ("reason-128", _wire((_record(0, reason=_LONG),))),
        ("pts-and-reason", _wire((_record(0, source_pts_ns=2**64 - 1, reason="score-missing"),))),
        ("parent", _wire((first, _record(1, parent_record_id=first.record_id)))),
        *[
            (f"observed-at-{ns}", _wire((_record(0, observed_at_ns=ns),)))
            for ns in (0, 2**63 - 1, 2**64 - 1)
        ],
        (
            "identities-128",
            _wire(
                (_record(0, producer=_LONG, causal_unit_id=_LONG, outcome=_LONG, **long_ids),),
                (_gap(producer=_LONG, cause=_LONG),),
                provenance=_provenance(**dict.fromkeys(WireProvenance.__slots__, _LONG)),
                **long_ids,
            ),
        ),
        *[(f"gap-{cause}", _wire((first,), (_gap(cause=cause),))) for cause in _GAP_CAUSES],
        ("gap-custom-cause", _wire(gaps=(_gap(cause="operator-reset"),))),
        ("gap-without-range", _wire(gaps=(_gap(source_generation=None, stream_epoch=None),))),
        ("gap-zero-count", _wire(gaps=(_gap(record_count=0),))),
        ("gaps-only", _wire(gaps=tuple(_gap(cause=cause) for cause in _GAP_CAUSES))),
        ("extra-batch-key", _edited(_wire((first,)), lambda b: b.update(schema="v2"))),
        (
            "extra-provenance-key",
            _edited(_wire((first,)), lambda b: b["provenance"].update(host="edge-1")),
        ),
        (
            "extra-gap-key",
            _edited(_wire(gaps=(_gap(),)), lambda b: b["gaps"][0].update(note="drained")),
        ),
    ]


def _set(path: tuple[str | int, ...], value: object) -> Edit:
    def edit(body: Body) -> None:
        target: Any = body
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return edit


def _drop(path: tuple[str | int, ...]) -> Edit:
    def edit(body: Body) -> None:
        target: Any = body
        for step in path[:-1]:
            target = target[step]
        del target[path[-1]]

    return edit


_REJECTED: list[tuple[str, Edit]] = [
    ("empty-camera", _set(("camera_id",), "")),
    ("camera-129", _set(("camera_id",), "x" * 129)),
    ("empty-producer", _set(("records", 0, "producer"), "")),
    ("empty-outcome", _set(("records", 0, "outcome"), "")),
    ("reason-empty", _set(("records", 0, "reason"), "")),
    ("reason-129", _set(("records", 0, "reason"), "x" * 129)),
    ("empty-provenance-field", _set(("provenance", "policy_identity"), "")),
    ("missing-provenance-field", _drop(("provenance", "policy_identity"))),
    ("missing-provenance", _drop(("provenance",))),
    ("negative-generation", _set(("records", 0, "source_generation"), -1)),
    ("negative-sequence", _set(("records", 0, "producer_sequence"), -1)),
    ("negative-observed-at", _set(("records", 0, "observed_at_ns"), -1)),
    ("negative-frame-seq", _set(("records", 0, "frame_seq"), -1)),
    ("negative-gap-count", _set(("gaps", 0, "record_count"), -1)),
    ("negative-gap-from-ns", _set(("gaps", 0, "from_ns"), -1)),
    ("empty-gap-cause", _set(("gaps", 0, "cause"), "")),
    ("uppercase-parent", _set(("records", 0, "parent_record_id"), "A" * 64)),
    ("unknown-kind", _set(("records", 0, "record_kind"), "sdk.unknown")),
    ("unknown-time-quality", _set(("records", 0, "time_quality"), "gps")),
    ("extra-record-key", _set(("records", 0, "note"), "x")),
]


def _rejectable() -> Body:
    record = _record(0)
    return _wire((record,), (_gap(),))


_ACCEPTED = _accepted_bodies()


@pytest.mark.parametrize(("name", "body"), _ACCEPTED, ids=[name for name, _ in _ACCEPTED])
def test_the_request_schema_accepts_every_body_the_wire_contract_accepts(
    name: str, body: Body
) -> None:
    WireBatch.from_json(body)
    ExecutionRecordBatchRequest.model_validate(body)


@pytest.mark.parametrize(("name", "edit"), _REJECTED, ids=[name for name, _ in _REJECTED])
def test_the_request_schema_rejects_what_the_wire_contract_rejects(name: str, edit: Edit) -> None:
    body = _edited(_rejectable(), edit)
    with pytest.raises(ExecutionRecordContractError):
        WireBatch.from_json(body)
    with pytest.raises(ValidationError):
        ExecutionRecordBatchRequest.model_validate(body)
