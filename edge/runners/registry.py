"""Task-to-runner registry for model execution.

Model swaps stay constrained to the runner implementation, artifact, and config
wiring. Callers select a task and receive the configured runner without
importing lower-level API modules directly.
"""

from __future__ import annotations

from collections.abc import Callable

from contracts.runner import RunnerProtocol
from edge.runners.sklearn_fall import FallDetector
from edge.runners.yolo_bed_seg import YoloBedSegRunner
from edge.runners.yolo_person import YoloPersonRunner
from edge.runners.yolo_pose import YoloPoseRunner

RunnerFactory = Callable[..., RunnerProtocol]


class ModelRegistry:
    """Small registry mapping model task names to runner factories."""

    def __init__(self) -> None:
        self._factories: dict[str, RunnerFactory] = {}

    def register(self, task: str, factory: RunnerFactory) -> None:
        if not task:
            raise ValueError("task must be non-empty")
        self._factories[task] = factory

    def create(self, task: str, **kwargs: object) -> RunnerProtocol:
        return self.get_factory(task)(**kwargs)

    def get_factory(self, task: str) -> RunnerFactory:
        try:
            return self._factories[task]
        except KeyError as exc:
            raise KeyError(f"unknown model task {task!r}") from exc

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("pose", YoloPoseRunner)
    registry.register("bed", YoloBedSegRunner)
    registry.register("person", YoloPersonRunner)
    registry.register("fall", FallDetector)
    return registry


DEFAULT_REGISTRY = default_registry()
