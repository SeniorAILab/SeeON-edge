from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from edge.runners.torch_lstm_fall import LstmFallManifest, ModelLoadError


def _write_manifest(path: Path) -> Path:
    payload = {
        "type": "lstm",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": str(path.parent),
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        "window": 3,
        "stride": 1,
        "input_shape": [3, 51],
        "operating_threshold": 0.5,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_lstm_manifest_accepts_metadata_yaml(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    (tmp_path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "metadata.yaml")

    manifest = LstmFallManifest.from_yaml(manifest_path)

    assert manifest.artifact_dir == tmp_path
    assert manifest.weights_path == tmp_path / "model.pt"
    assert manifest.architecture_path == tmp_path / "arch.json"
    assert manifest.metadata_path == tmp_path / "metadata.yaml"
    assert manifest.input_shape == (3, 51)


def test_lstm_manifest_rejects_metadata_json_fallback(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="metadata.yaml"):
        LstmFallManifest.from_artifact_dir(tmp_path)


def test_lstm_manifest_rejects_wrong_input_shape(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    (tmp_path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "metadata.yaml")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["input_shape"] = [3, 45]
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="input_shape"):
        LstmFallManifest.from_yaml(manifest_path)
