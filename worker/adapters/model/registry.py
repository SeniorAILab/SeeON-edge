from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol, TypeAlias

from contracts.runner import RunnerProtocol
from worker.interfaces.fall_model import FallModelProtocol

ModelOption: TypeAlias = str | int | float | bool | None


class FallModel(FallModelProtocol, Protocol):
    def warmup(self) -> None: ...


class WarmupModel(Protocol):
    def warmup(self) -> None: ...


ModelAdapter: TypeAlias = RunnerProtocol
RunnerFactory: TypeAlias = Callable[..., ModelAdapter]


class EmptyModelTaskError(ValueError):
    def __init__(self) -> None:
        super().__init__("task must be non-empty")


class UnknownModelTaskError(KeyError):
    task: str

    def __init__(self, task: str) -> None:
        self.task = task
        super().__init__(f"unknown model task {task!r}")


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, RunnerFactory] = {}

    def register(self, task: str, factory: RunnerFactory) -> None:
        if not task:
            raise EmptyModelTaskError
        self._factories[task] = factory

    def create(self, task: str, **kwargs: ModelOption) -> ModelAdapter:
        return self.get_factory(task)(**kwargs)

    def get_factory(self, task: str) -> RunnerFactory:
        try:
            return self._factories[task]
        except KeyError as exc:
            raise UnknownModelTaskError(task) from exc

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("pose", _yolo_pose)
    registry.register("person", _yolo_person)
    registry.register("bed", _yolo_bed_seg)
    return registry


def _yolo_pose(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_pose import YoloPoseRunner

    return YoloPoseRunner(**kwargs)


def _yolo_person(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_person import YoloPersonRunner

    return YoloPersonRunner(**kwargs)


def _yolo_bed_seg(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner

    return YoloBedSegRunner(**kwargs)


def flow_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("bed", _ort_bed_seg)
    return registry


def _ort_bed_seg(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.ort_bed_seg import OrtBedSegRunner

    return OrtBedSegRunner(**kwargs)


DEFAULT_REGISTRY: Final = default_registry()

__all__ = [
    "DEFAULT_REGISTRY",
    "EmptyModelTaskError",
    "FallModel",
    "ModelAdapter",
    "ModelOption",
    "ModelRegistry",
    "RunnerFactory",
    "UnknownModelTaskError",
    "WarmupModel",
    "default_registry",
    "flow_registry",
]
