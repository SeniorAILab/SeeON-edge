"""Immutable snapshot of one named frame-bus subscription's counters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BusMetricsSnapshot:
    """Point-in-time counters for a single named subscription."""

    published: int
    taken: int
    dropped: int
    queue_age_sec: float


__all__ = ["BusMetricsSnapshot"]
