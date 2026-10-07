from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worker.adapters.model.registry import EmptyModelTaskError, ModelRegistry, default_registry
from worker.adapters.model.sklearn_fall import MODELS_DIR


class FakeRunner:
    def __init__(self, *, value: int = 0) -> None:
        self.value = value


def test_model_registry_rejects_empty_task() -> None:
    registry = ModelRegistry()

    with pytest.raises(EmptyModelTaskError, match="task must be non-empty"):
        registry.register("", FakeRunner)


def test_default_registry_has_pose_bed_person_factories_without_loading_models() -> None:
    registry = default_registry()

    assert registry.tasks() == ("bed", "person", "pose")
    for task in ("pose", "bed", "person"):
        assert callable(registry.get_factory(task))
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from worker.adapters.model.registry import default_registry; "
                "default_registry(); "
                "print(sorted(m for m in sys.modules "
                "if m.startswith('worker.adapters.model.yolo_')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "[]"


def test_sklearn_fall_default_models_dir_points_to_ml_models_root() -> None:
    assert Path(__file__).resolve().parents[1] / "models" == MODELS_DIR
