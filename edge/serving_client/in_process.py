"""In-process serving client: a pure pass-through over the local ModelRegistry.

This is the identity implementation of the serving seam — it provisions runners
directly in the edge process (no network, identical behavior to calling the
registry directly). The ``ServingClient`` contract is evolving toward batch-input
(frame-list) inference so a future networked batched serving service (50-camera
scale) can implement the same interface without a caller rewrite (see ADR-0002).
"""

from __future__ import annotations

from contracts.runner import RunnerProtocol
from edge.runners.registry import DEFAULT_REGISTRY, ModelRegistry


class InProcessServingClient:
    """Pass-through ``ServingClient`` backed by a local :class:`ModelRegistry`."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = DEFAULT_REGISTRY if registry is None else registry

    def create(self, task: str, **kwargs: object) -> RunnerProtocol:
        return self._registry.create(task, **kwargs)
