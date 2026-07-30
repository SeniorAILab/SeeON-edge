from __future__ import annotations

from pathlib import Path

import pytest

from edge.runners.registry import ModelRegistry, default_registry
from edge.runners.sklearn_fall import MODELS_DIR, FallDetector
from edge.runners.yolo_bed_seg import YoloBedSegRunner
from edge.runners.yolo_person import YoloPersonRunner
from edge.runners.yolo_pose import YoloPoseRunner


class FakeRunner:
    def __init__(self, *, value: int = 0) -> None:
        self.value = value


def test_model_registry_register_create_get_factory_and_tasks() -> None:
    registry = ModelRegistry()
    registry.register("fake", FakeRunner)

    assert registry.tasks() == ("fake",)
    assert registry.get_factory("fake") is FakeRunner
    runner = registry.create("fake", value=7)

    assert isinstance(runner, FakeRunner)
    assert runner.value == 7


def test_model_registry_rejects_empty_task() -> None:
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="task must be non-empty"):
        registry.register("", FakeRunner)


def test_model_registry_unknown_task_raises() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="unknown model task"):
        registry.get_factory("missing")

    with pytest.raises(KeyError, match="unknown model task"):
        registry.create("missing")


def test_default_registry_has_pose_bed_person_fall_factories_without_loading_models() -> None:
    registry = default_registry()

    assert registry.tasks() == ("bed", "fall", "person", "pose")
    assert registry.get_factory("pose") is YoloPoseRunner
    assert registry.get_factory("bed") is YoloBedSegRunner
    assert registry.get_factory("person") is YoloPersonRunner
    assert registry.get_factory("fall") is FallDetector

def test_sklearn_fall_default_models_dir_points_to_ml_models_root() -> None:
    assert MODELS_DIR == Path(__file__).resolve().parents[1] / "models"
