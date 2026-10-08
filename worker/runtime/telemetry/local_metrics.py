from __future__ import annotations

import logging
from typing import final

from worker.runtime.telemetry.models import (
    BusMetricsSource,
    BusSubscriptionSnapshot,
    RuntimeDiagnosticsSnapshot,
    StageTimingSnapshot,
)

LOGGER = logging.getLogger(__name__)


@final
class StageTimingAccumulator:
    __slots__ = ("last_sec", "max_sec", "samples", "total_sec")

    def __init__(self) -> None:
        self.samples = 0
        self.total_sec = 0.0
        self.last_sec = 0.0
        self.max_sec = 0.0

    def add(self, elapsed_sec: float) -> None:
        self.samples += 1
        self.total_sec += elapsed_sec
        self.last_sec = elapsed_sec
        self.max_sec = max(self.max_sec, elapsed_sec)

    def snapshot(self, stage: str) -> StageTimingSnapshot:
        return StageTimingSnapshot(
            stage=stage,
            samples=self.samples,
            total_sec=self.total_sec,
            last_sec=self.last_sec,
            max_sec=self.max_sec,
        )


def bus_snapshot(
    source: tuple[BusMetricsSource, tuple[str, ...]] | None,
) -> tuple[BusSubscriptionSnapshot, ...]:
    if source is None:
        return ()
    bus, names = source
    return tuple(
        BusSubscriptionSnapshot(
            name=name,
            published=(metrics := bus.metrics(name)).published,
            taken=metrics.taken,
            dropped=metrics.dropped,
            queue_age_sec=metrics.queue_age_sec,
        )
        for name in names
    )


def log_snapshot(snapshot: RuntimeDiagnosticsSnapshot) -> None:
    for camera in snapshot.cameras:
        inference = camera.inference
        fields = [
            f"camera_id={camera.camera_id}",
            f"failure_category={camera.failure_category}",
            f"inferred={0 if inference is None else inference.inferred}",
            f"bus_dropped={sum(metrics.dropped for metrics in camera.bus)}",
            f"forward_p95_sec={camera.forward_p95_sec:.3f}",
        ]
        if camera.decode_backend is not None:
            fields.append(f"decode_backend={camera.decode_backend.resolved_backend}")
        if camera.fall_unapplied_policy_threshold is not None:
            fields.append(
                f"fall_unapplied_policy_threshold={camera.fall_unapplied_policy_threshold}"
            )
        LOGGER.info("worker.runtime.telemetry %s", " ".join(fields))


__all__ = ["StageTimingAccumulator", "bus_snapshot", "log_snapshot"]
