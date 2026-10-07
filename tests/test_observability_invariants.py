from __future__ import annotations

import inspect
import time

import psycopg
import pytest
from test_execution_record_store import (
    CAMERA,
    _batch,
    _record,
    _store,
)

from backend.app.features.diagnostics import retention as retention_module
from backend.app.features.diagnostics.coverage import availability
from backend.app.features.diagnostics.records import AvailabilityKind
from backend.app.features.diagnostics.retention import (
    MAX_UNITS_PER_ENFORCE,
    RetentionBudget,
    enforce_budget,
    used_bytes,
)
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox
from worker.pipeline.diagnostics import emit_delivery, emit_policy

pytest_plugins = ("tests_support.postgres_diagnostics_sandbox",)

_YEAR_2020_NS = 1_577_836_800_000_000_000


def _record_builders() -> dict[str, object]:
    return {
        name: value
        for module in (emit_policy, emit_delivery)
        for name, value in vars(module).items()
        if (
            name.endswith("_record")
            and inspect.isfunction(value)
            and value.__module__ == module.__name__
        )
    }


_WALL_STAMPED_BUILDERS = frozenset(
    {
        "sdk_frame_record",
        "policy_consume_record",
        "model_score_record",
        "policy_decision_record",
        "policy_coast_record",
        "event_delivery_record",
        "delivery_attempt_record",
        "backend_acceptance_record",
    }
)


def test_record_builder_set_is_closed() -> None:
    assert set(_record_builders()) == set(_WALL_STAMPED_BUILDERS)


def test_every_record_builder_stamps_wall_clock() -> None:
    before = time.time_ns()
    records = [
        emit_policy.policy_coast_record(
            camera_id=CAMERA,
            worker_boot_id="boot-1",
            source_generation=1,
            stream_epoch=1,
            frame_seq=1,
            source_pts_ns=None,
            module_qualified_id="fall.v2",
        ),
        emit_delivery.delivery_attempt_record(
            camera_id=CAMERA,
            observing_boot_id="boot-1",
            edge_event_id="event-1",
            outcome="retry-transient",
            attempt=1,
            max_attempts=10,
            failure_class="RETRY",
            status_code=503,
            retained=None,
        ),
        emit_delivery.backend_acceptance_record(
            camera_id=CAMERA,
            observing_boot_id="boot-1",
            edge_event_id="event-1",
            status="accepted_local",
            hub_event_id=None,
        ),
    ]
    after = time.time_ns()
    assert all(record is not None for record in records)
    for record in records:
        assert record.time_quality == "wall", record.record_kind
        assert before <= record.observed_at_ns <= after, record.record_kind
        assert record.observed_at_ns > _YEAR_2020_NS, record.record_kind


def test_availability_range_count_scales_with_gaps_not_records(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    budget = RetentionBudget(total_bytes=8 * 2**20)
    del budget
    store = _store(diag, total_bytes=8 * 2**20)
    first = tuple(
        _record(label=f"a{index}", seq=index, observed=1_000 + index * 33) for index in range(600)
    )
    second = tuple(
        _record(label=f"b{index}", seq=index, observed=1_000 + index * 33)
        for index in range(700, 1_000)
    )
    store.ingest_batch(_batch("run-a", first))
    store.ingest_batch(_batch("run-b", second))

    painted = diag.database.read_snapshot(
        lambda connection: availability(connection, CAMERA, 0, 2_000 * 33)
    )

    kinds = [item.kind for item in painted]
    available = [item for item in painted if item.kind is AvailabilityKind.AVAILABLE]
    assert len(available) == 2, kinds
    assert len(painted) <= 8, f"{len(painted)} ranges for 900 records: {kinds}"
    assert AvailabilityKind.AVAILABLE in kinds


def test_enforce_budget_does_bounded_work_however_deep_the_backlog(
    postgres_diagnostics_sandbox: DiagnosticsSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag = postgres_diagnostics_sandbox
    budget = RetentionBudget(total_bytes=128 * 1024, unit_horizon_ns=1_000)
    store = _store(diag, total_bytes=1 << 40)
    blob = {"blob": "x" * 200}
    for unit in range(60):
        store.ingest_batch(
            _batch(
                f"fill-{unit}",
                tuple(
                    _record(
                        label=f"u{unit}-{index}",
                        unit=f"unit-{unit:02d}",
                        seq=unit * 100 + index,
                        observed=10 + unit * 5 + index,
                        payload=blob,
                    )
                    for index in range(30)
                ),
            )
        )

    walks = 0
    real_used_bytes = retention_module.used_bytes

    def _counting_used_bytes(connection: psycopg.Connection) -> int:
        nonlocal walks
        walks += 1
        return real_used_bytes(connection)

    monkeypatch.setattr(retention_module, "used_bytes", _counting_used_bytes)

    admin = diag.admin
    assert used_bytes(admin) > budget.high_water
    before = admin.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
    committed = diag.database.transact(
        lambda connection: enforce_budget(connection, budget, 10_000_000)
    )
    after = admin.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]

    assert committed is True, "progress was made, so the ingest may commit"
    assert before - after <= MAX_UNITS_PER_ENFORCE, "one call pruned an unbounded backlog"
    assert before - after >= 1, "a call that commits must make progress"
    assert walks <= 2, f"{walks} used_bytes walks in one call"
