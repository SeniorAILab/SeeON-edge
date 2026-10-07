from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import psycopg

from backend.app.features.diagnostics.prune import (
    Lane,
    coarsen_coverage,
    drop_orphan_batches,
    next_prunable_unit,
    prune_unit,
)
from backend.app.features.diagnostics.terminals import (
    force_oldest_units_terminal,
    refresh_unit_terminals,
)

COVERAGE_ROWS_PER_EPOCH: Final = 512
DEFAULT_UNIT_HORIZON_NS: Final = 60_000_000_000


@dataclass(frozen=True, slots=True)
class RetentionBudget:
    total_bytes: int
    unit_horizon_ns: int = DEFAULT_UNIT_HORIZON_NS
    coverage_rows_per_epoch: int = COVERAGE_ROWS_PER_EPOCH

    def __post_init__(self) -> None:
        if type(self.total_bytes) is not int or self.total_bytes < 256:
            raise ValueError(
                "total_bytes must be an explicit integer >= 256 "
                "(max_record_bytes = total_bytes // 256 must be >= 1; "
                "this floor is not a deployment size — it only bounds the "
                "largest accepted record)"
            )
        if type(self.unit_horizon_ns) is not int or self.unit_horizon_ns <= 0:
            raise ValueError("unit_horizon_ns must be a positive integer")
        if type(self.coverage_rows_per_epoch) is not int or self.coverage_rows_per_epoch < 1:
            raise ValueError("coverage_rows_per_epoch must be a positive integer")

    @property
    def control_reserve(self) -> int:
        return self.total_bytes // 16

    @property
    def high_water(self) -> int:
        return self.total_bytes - self.control_reserve

    @property
    def low_water(self) -> int:
        return (self.high_water * 7) // 8

    @property
    def segment_bytes(self) -> int:
        return self.total_bytes // 64

    @property
    def max_record_bytes(self) -> int:
        return self.total_bytes // 256


def used_bytes(connection: psycopg.Connection) -> int:
    row = connection.execute(
        """
        SELECT
            (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_provenance AS t)
          + (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_segments AS t)
          + (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_units AS t)
          + (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_records AS t)
          + (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_coverage AS t)
          + (SELECT COALESCE(SUM(pg_column_size(t.*)), 0) FROM execution_batches AS t)
        """
    ).fetchone()
    return 0 if row is None else int(row[0])


MAX_UNITS_PER_ENFORCE: Final = 32

USAGE_REMEASURE_NS: Final = 1_000_000_000


class UsageMeter:
    __slots__ = ("_accrued", "_measured", "_measured_at", "_ratio", "_remeasure_ns")

    def __init__(self, *, remeasure_ns: int = USAGE_REMEASURE_NS) -> None:
        self._remeasure_ns = remeasure_ns
        self._measured: int | None = None
        self._measured_at = 0
        self._ratio = 1.0
        self._accrued = 0

    def accrue(self, payload_bytes: int) -> None:
        self._accrued += int(max(0, payload_bytes) * self._ratio)

    def invalidate(self) -> None:
        self._measured = None

    def release(self, freed_payload_bytes: int) -> int:
        if self._measured is None:
            return 0
        self._accrued -= int(max(0, freed_payload_bytes) * self._ratio)
        return max(0, self._measured + self._accrued)

    def measured_over(self, high_water: int) -> bool:
        return self._measured is not None and self._measured > high_water

    def value(self, connection: psycopg.Connection, now_ns: int) -> int:
        stale = self._measured is None or now_ns - self._measured_at >= self._remeasure_ns
        if stale:
            self._measured = used_bytes(connection)
            self._measured_at = now_ns
            self._accrued = 0
            payload = connection.execute(
                "SELECT COALESCE(SUM(payload_bytes), 0) FROM execution_segments"
            ).fetchone()
            payload_total = int(payload[0]) if payload else 0
            self._ratio = self._measured / payload_total if payload_total > 0 else 1.0
        return self._measured + self._accrued


def enforce_budget(
    connection: psycopg.Connection,
    budget: RetentionBudget,
    now_ns: int,
    *,
    max_units: int = MAX_UNITS_PER_ENFORCE,
    meter: UsageMeter | None = None,
) -> bool:
    gauge = meter if meter is not None else UsageMeter()
    refresh_unit_terminals(connection, budget.unit_horizon_ns)
    occupied = gauge.value(connection, now_ns)
    touched: set[Lane] = set()
    if occupied > budget.high_water and not gauge.measured_over(budget.high_water):
        gauge.invalidate()
        occupied = gauge.value(connection, now_ns)
    pruned = 0
    while occupied > budget.high_water and pruned < max_units:
        unit_id = next_prunable_unit(connection)
        if unit_id is None:
            if force_oldest_units_terminal(connection, 1) == 0:
                break
            unit_id = next_prunable_unit(connection)
            if unit_id is None:
                break
        freed_payload, lane = prune_unit(connection, unit_id, now_ns)
        if lane is not None:
            touched.add(lane)
        pruned += 1
        occupied = gauge.release(freed_payload)
        if occupied <= budget.low_water:
            break
    if pruned:
        coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns, touched)
        drop_orphan_batches(connection)
        return True
    return occupied <= budget.high_water


__all__ = [
    "COVERAGE_ROWS_PER_EPOCH",
    "DEFAULT_UNIT_HORIZON_NS",
    "MAX_UNITS_PER_ENFORCE",
    "USAGE_REMEASURE_NS",
    "RetentionBudget",
    "UsageMeter",
    "coarsen_coverage",
    "enforce_budget",
    "force_oldest_units_terminal",
    "next_prunable_unit",
    "prune_unit",
    "refresh_unit_terminals",
    "used_bytes",
]
