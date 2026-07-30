"""Serving-client seam: the boundary edge uses to obtain model runners.

Keeping this interface between the edge pipeline and the concrete model runners
lets the GPU inference plane be extracted into a separate serving service later
(swap ``InProcessServingClient`` for a networked client) without touching the
edge orchestration/perception/domain code.

``BatchServingClient`` is the evolution-ready extension of this seam for the
50-camera scale target (ADR-0002): a future networked serving service batches
frames across cameras into one GPU forward pass behind the same seam, so
per-camera callers need no rewrite. The batching backend is intentionally NOT
implemented in-process (deferred, ADR-0002); this is the documented, typed swap
point only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contracts.runner import RunnerProtocol


@runtime_checkable
class ServingClient(Protocol):
    """Provisions model runners by task name (pose/bed/fall/...)."""

    def create(self, task: str, **kwargs: object) -> RunnerProtocol:
        """Return a runner for ``task``; kwargs (e.g. ``device``) pass through."""
        ...


@runtime_checkable
class BatchServingClient(ServingClient, Protocol):
    """Evolution-ready seam: cross-camera batched inference for 50-camera scale.

    Defines the batched-inference contract a future networked serving service
    implements behind the same seam (ADR-0002). ``frames`` is one micro-batch
    (typically one frame per active camera); the result is aligned per input.
    The in-process client does not implement this yet — the batching backend is
    deferred per ADR-0002 — but the contract is fixed so callers can adopt it
    without a later rewrite.
    """

    def infer_batch(
        self, task: str, frames: Sequence[object], **kwargs: object
    ) -> list[object]:
        """Run ``task`` over a batch of frames; return one result per input frame."""
        ...
