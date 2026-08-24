"""Binary differential comparator: native strategy vs the Python tracker oracle.

Parity is exact, not approximate. Per-frame `track_ids` and the `live_ids`
snapshot after every call must match byte/value; the first divergence names
the frame index and field so a reviewer can tell an ordering bug from an
eviction bug from a tie-break bug. `compare_traces` returns an empty tuple
only when every frame agreed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssociationParityMismatch:
    """One binary parity failure, named by the frame and field that diverged."""

    frame_index: int
    field: str
    reference: object
    candidate: object

    def __str__(self) -> str:
        return (
            f"frame[{self.frame_index}].{self.field}: "
            f"reference={self.reference!r} candidate={self.candidate!r}"
        )


@dataclass(frozen=True, slots=True)
class AssociationFrameTrace:
    """One frame's observable association outcome, from either implementation."""

    track_ids: tuple[int, ...]
    live_ids: frozenset[int]


def compare_traces(
    reference: tuple[AssociationFrameTrace, ...],
    candidate: tuple[AssociationFrameTrace, ...],
) -> tuple[AssociationParityMismatch, ...]:
    """Compare two full per-frame traces, one frame at a time, in order."""
    if len(reference) != len(candidate):
        return (
            AssociationParityMismatch(
                frame_index=-1,
                field="frame_count",
                reference=len(reference),
                candidate=len(candidate),
            ),
        )
    mismatches: list[AssociationParityMismatch] = []
    for frame_index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if expected.track_ids != actual.track_ids:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="track_ids",
                    reference=expected.track_ids,
                    candidate=actual.track_ids,
                )
            )
        if expected.live_ids != actual.live_ids:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="live_ids",
                    reference=expected.live_ids,
                    candidate=actual.live_ids,
                )
            )
    return tuple(mismatches)


__all__ = ["AssociationFrameTrace", "AssociationParityMismatch", "compare_traces"]
