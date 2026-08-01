"""In-process implementation of the worker model-serving seam."""

from __future__ import annotations

from worker.adapters.model.registry import (
    DEFAULT_REGISTRY,
    ModelAdapter,
    ModelOption,
    ModelRegistry,
)


class InProcessServingClient:
    """Delegate single-model provisioning to a process-local registry."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry: ModelRegistry = DEFAULT_REGISTRY if registry is None else registry

    def create(self, task: str, **kwargs: ModelOption) -> ModelAdapter:
        return self._registry.create(task, **kwargs)


__all__ = ["InProcessServingClient"]
