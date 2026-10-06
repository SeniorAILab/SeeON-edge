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
        stage_timings = {
            timing.stage: {
                "samples": timing.samples,
                "total_sec": timing.total_sec,
                "last_sec": timing.last_sec,
                "max_sec": timing.max_sec,
            }
            for timing in camera.stage_timings
        }
        bus = {
            metrics.name: {
                "published": metrics.published,
                "taken": metrics.taken,
                "dropped": metrics.dropped,
                "queue_age_sec": metrics.queue_age_sec,
            }
            for metrics in camera.bus
        }
        decode_backend = (
            None
            if camera.decode_backend is None
            else {
                "requested_profile_decode": (camera.decode_backend.requested_profile_decode),
                "resolved_backend": camera.decode_backend.resolved_backend,
                "actual_adapter_class": camera.decode_backend.actual_adapter_class,
            }
        )
        encode = (
            None
            if camera.encode is None
            else {
                "requested": camera.encode.requested,
                "selected": camera.encode.selected,
                "fallback_count": camera.encode.fallback_count,
                "last_reason": camera.encode.last_reason,
            }
        )
        bed_region = (
            None
            if camera.bed_region is None
            else {
                "freshness": camera.bed_region.freshness.value,
                "counters": dict(camera.bed_region.counters),
                "updated_at_sec": camera.bed_region.updated_at_sec,
            }
        )
        bed_exit_scoring = (
            None
            if camera.bed_exit_scoring is None
            else {
                "max_containment_observed": (camera.bed_exit_scoring.max_containment_observed),
                "grace_positive_transitions": (camera.bed_exit_scoring.grace_positive_transitions),
                "assignments_made": camera.bed_exit_scoring.assignments_made,
                "updated_at_sec": camera.bed_exit_scoring.updated_at_sec,
            }
        )
        observed = (
            None
            if camera.inference is None or camera.inference.observed_geometry is None
            else f"{camera.inference.observed_geometry[0]}x{camera.inference.observed_geometry[1]}"
        )
        geometry_batch_sizes = {
            f"{histogram.geometry[0]}x{histogram.geometry[1]}": dict(histogram.batch_sizes)
            for histogram in camera.geometry_batch_sizes
        }
        inference = (
            None
            if camera.inference is None
            else {
                "admitted": camera.inference.admitted,
                "overwritten": camera.inference.overwritten,
                "inferred": camera.inference.inferred,
                "queue_age_sec": camera.inference.queue_age_sec,
                "observed_geometry": observed,
                "batch_sizes": dict(camera.batch_sizes),
                "geometry_batch_sizes": geometry_batch_sizes,
                "forward_p50_sec": camera.forward_p50_sec,
                "forward_p95_sec": camera.forward_p95_sec,
            }
        )
        LOGGER.info(
            "worker.runtime.telemetry camera_id=%s failure_category=%s "
            "stage_timings=%s bus=%s encoder=%s encode=%s bed_region=%s "
            "bed_exit_scoring=%s inference=%s"
            " smart_record_extended_total=%d smart_record_extension_raced_total=%d"
            " smart_record_start_refused_total=%d nvenc_sessions_active=%d"
            + (" decode_backend=%s" if decode_backend is not None else "")
            + (
                " fall_unapplied_policy_threshold=%s"
                if camera.fall_unapplied_policy_threshold is not None
                else ""
            ),
            camera.camera_id,
            camera.failure_category,
            stage_timings,
            bus,
            snapshot.encoder,
            encode,
            bed_region,
            bed_exit_scoring,
            inference,
            camera.smart_record_extended_total,
            camera.smart_record_extension_raced_total,
            camera.smart_record_start_refused_total,
            camera.nvenc_sessions_active,
            *((decode_backend,) if decode_backend is not None else ()),
            *(
                (camera.fall_unapplied_policy_threshold,)
                if camera.fall_unapplied_policy_threshold is not None
                else ()
            ),
            extra={
                "camera_id": camera.camera_id,
                "failure_category": camera.failure_category,
                "stage_timings": stage_timings,
                "bus": bus,
                "encoder": snapshot.encoder,
                "decode_backend": decode_backend,
                "bed_region": bed_region,
                "bed_exit_scoring": bed_exit_scoring,
                "inference": inference,
                "fall_unapplied_policy_threshold": camera.fall_unapplied_policy_threshold,
            },
        )


__all__ = ["StageTimingAccumulator", "bus_snapshot", "log_snapshot"]
