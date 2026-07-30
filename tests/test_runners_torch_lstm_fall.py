from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import edge.runners.torch_lstm_fall as torch_lstm_fall
from edge.runners.torch_lstm_fall import (
    LstmFallRunner,
    ModelLoadError,
    build_lstm_module,
)


def _write_lstm_artifact(path: Path) -> Path:

    path.mkdir()
    arch = {"hidden": 4, "layers": 1, "dropout": 0.0}
    module = build_lstm_module(hidden=4, layers=1, dropout=0.0)
    with torch.no_grad():
        module.fc.bias[0] = -5.0
        module.fc.bias[1] = 5.0
    torch.save(module.state_dict(), path / "model.pt")
    (path / "arch.json").write_text(json.dumps(arch), encoding="utf-8")
    (path / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "lstm",
                "framework": "pytorch",
                "mode": "sequence",
                "artifact_dir": str(path),
                "schema_version": 2,
                "preprocessing_identity": "coco17-xyc-frame-normalized-zero-fill-v1",
                "weights": "model.pt",
                "architecture": "arch.json",
                "metadata": "metadata.yaml",
                "window": 3,
                "stride": 1,
                "input_shape": [3, 51],
                "operating_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path

def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        window=4,
        stride=1,
        mode="sequence",
        operating_threshold=0.5,
        input_shape=(4, 51),
        schema_version=2,
        preprocessing_identity="coco17-xyc-frame-normalized-zero-fill-v1",
    )


def test_runner_defaults_to_cpu() -> None:
    runner = LstmFallRunner(
        _manifest(), build_lstm_module(hidden=8, layers=1, dropout=0.0)
    )

    probability = runner.predict(np.zeros((4, 51), dtype=np.float32))

    assert runner.device == torch.device("cpu")
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_predict_moves_input_to_model_device() -> None:
    runner = LstmFallRunner(
        _manifest(), build_lstm_module(hidden=8, layers=1, dropout=0.0), device="cpu"
    )

    probability = runner.predict(np.zeros((4, 51), dtype=np.float32))

    assert next(runner.model.parameters()).device.type == "cpu"
    assert 0.0 <= probability <= 1.0
def test_lstm_runner_moves_module_and_input_to_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_device = torch.device("meta")
    module_devices: list[torch.device] = []
    input_devices: list[torch.device] = []

    class FakeModule:
        def to(self, device: torch.device) -> FakeModule:
            module_devices.append(device)
            return self

        def eval(self) -> FakeModule:
            return self

        def __call__(self, tensor: object) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0]])

    class FakeInput:
        def unsqueeze(self, _: int) -> FakeInput:
            return self

        def to(self, device: torch.device) -> FakeInput:
            input_devices.append(device)
            return self

    monkeypatch.setattr(torch_lstm_fall.torch, "from_numpy", lambda _: FakeInput())
    runner = LstmFallRunner(_manifest(), FakeModule(), device=str(requested_device))

    assert runner.predict(np.zeros((4, 51), dtype=np.float32)) > 0.5
    assert module_devices == [requested_device]
    assert input_devices == [requested_device]



def test_from_artifact_dir_threads_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    manifest.architecture_path = tmp_path / "arch.json"
    manifest.weights_path = tmp_path / "model.pt"
    module = build_lstm_module(hidden=8, layers=1, dropout=0.0)
    monkeypatch.setattr(
        torch_lstm_fall.LstmFallManifest,
        "from_artifact_dir",
        lambda artifact_dir: manifest,
    )
    monkeypatch.setattr(
        torch_lstm_fall, "build_lstm_module_from_arch", lambda path: module
    )
    monkeypatch.setattr(torch_lstm_fall, "_load_state_dict", lambda path: module.state_dict())

    runner = LstmFallRunner.from_artifact_dir(tmp_path, device="cpu")

    assert runner.device == torch.device("cpu")
    assert next(runner.model.parameters()).device.type == "cpu"


def test_place_on_bad_device_raises() -> None:
    with pytest.raises(ModelLoadError, match="cannot place LSTM fall model"):
        LstmFallRunner(
            _manifest(),
            build_lstm_module(hidden=8, layers=1, dropout=0.0),
            device="definitely-not-a-device",
        )


def test_lstm_runner_loads_generated_artifact_and_predicts_probability(tmp_path: Path) -> None:
    runner = LstmFallRunner.from_artifact_dir(_write_lstm_artifact(tmp_path / "lstm"))
    sequence = np.zeros((3, 51), dtype=np.float32)

    probability = runner.predict(sequence)

    assert 0.99 <= probability <= 1.0
    assert runner.metadata.window == 3
    assert runner.metadata.mode == "sequence"


def test_lstm_runner_rejects_wrong_input_shape(tmp_path: Path) -> None:
    runner = LstmFallRunner.from_artifact_dir(_write_lstm_artifact(tmp_path / "lstm"))

    with pytest.raises(ModelLoadError, match="input shape"):
        runner.predict(np.zeros((45,), dtype=np.float32))
