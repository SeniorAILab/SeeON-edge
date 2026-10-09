from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias

from typing_extensions import override

from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.registry import FallModel


class FallModelConfigLike(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def framework(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def artifact_dir(self) -> Path: ...

    @property
    def weights(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def metadata(self) -> str: ...

    @property
    def window(self) -> int: ...

    @property
    def stride(self) -> int: ...

    @property
    def input_shape(self) -> tuple[int, int]: ...

    @property
    def operating_threshold(self) -> float: ...

    @property
    def schema_version(self) -> int | None: ...

    @property
    def preprocessing_identity(self) -> str | None: ...


FallModelFactory: TypeAlias = Callable[[FallModelConfigLike, str], FallModel]


@dataclass(slots=True)
class UnknownFallModelTypeError(Exception):
    requested_type: str
    known_types: tuple[str, ...]

    @override
    def __str__(self) -> str:
        known = ", ".join(self.known_types) if self.known_types else "(none registered)"
        return (
            f"unknown fall model type {self.requested_type!r}; known fall model families: {known}"
        )


class FallModelFamilyRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, FallModelFactory] = {}

    def register(self, model_type: str, factory: FallModelFactory) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        self._factories[model_type] = factory

    def create(self, model_type: str, config: FallModelConfigLike, device: str) -> FallModel:
        return self.get_factory(model_type)(config, device)

    def get_factory(self, model_type: str) -> FallModelFactory:
        factory = self._factories.get(model_type)
        if factory is None:
            raise UnknownFallModelTypeError(model_type, self.types())
        return factory

    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _create_pose_bbox56_bundle_model(artifact_dir: Path, device: str) -> FallModel:
    from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner

    return PoseBbox56BundleRunner.from_artifact_dir(artifact_dir, device=device)


def _create_ort_pose_bbox56_bundle_model(artifact_dir: Path, device: str) -> FallModel:
    return OrtPoseBbox56Runner.from_artifact_dir(artifact_dir)


def _create_pose_bbox56_model(config: FallModelConfigLike, device: str) -> FallModel:
    if config.framework == "onnxruntime":
        return _create_ort_pose_bbox56_bundle_model(config.artifact_dir, device)
    return _create_pose_bbox56_bundle_model(config.artifact_dir, device)


def default_fall_model_family_registry() -> FallModelFamilyRegistry:
    registry = FallModelFamilyRegistry()
    registry.register("pose-bbox56-proxy-v0", _create_pose_bbox56_model)
    return registry


DEFAULT_FALL_MODEL_FAMILY_REGISTRY: Final = default_fall_model_family_registry()

__all__ = [
    "DEFAULT_FALL_MODEL_FAMILY_REGISTRY",
    "FallModelConfigLike",
    "FallModelFactory",
    "FallModelFamilyRegistry",
    "UnknownFallModelTypeError",
    "default_fall_model_family_registry",
]
