from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worker.adapters.model.errors import ModelLoadError
from worker.runtime.provenance.model_bundle import admit_model_bundle


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _record(path: str, content: bytes) -> dict[str, object]:
    return {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def _bundle(tmp_path: Path, *, receipts: dict[str, bytes] | None = None) -> Path:
    members = {"model.onnx": b"model", "calibration.json": b'{"calibration": true}'}
    root = tmp_path / "bundle"
    root.mkdir()
    for path, content in {**members, **(receipts or {})}.items():
        (root / path).write_bytes(content)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "runtime_format": "onnxruntime",
        "members": [_record(path, content) for path, content in members.items()],
    }
    if receipts:
        manifest["receipts"] = [_record(path, content) for path, content in receipts.items()]
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return root


def test_admission_returns_an_immutable_digest_map(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    digests = admit_model_bundle(root)
    assert dict(digests) == {
        "model.onnx": hashlib.sha256(b"model").hexdigest(),
        "calibration.json": hashlib.sha256(b'{"calibration": true}').hexdigest(),
    }
    with pytest.raises(TypeError):
        digests["model.onnx"] = "x"  # type: ignore[index]


def test_admission_rejects_extra_bundle_member(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    (root / "unexpected").write_text("unexpected")
    with pytest.raises(ModelLoadError, match="file tree differs"):
        admit_model_bundle(root)


def test_admission_requires_onnxruntime_runtime_format(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["runtime_format"] = "different-format"
    (root / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(ModelLoadError, match="is not onnxruntime"):
        admit_model_bundle(root)


def test_admission_rejects_non_canonical_manifest(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_bytes())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pytest.raises(ModelLoadError, match="not canonical JSON"):
        admit_model_bundle(root)


def test_admission_rejects_tampered_member_bytes(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    (root / "model.onnx").write_bytes(b"swapped")
    with pytest.raises(ModelLoadError, match="declared hash"):
        admit_model_bundle(root)


def test_admission_hash_checks_receipts_and_includes_them(tmp_path: Path) -> None:
    root = _bundle(tmp_path, receipts={"evaluation-receipt.json": b"not json"})
    assert "evaluation-receipt.json" in admit_model_bundle(root)
    (root / "evaluation-receipt.json").write_bytes(b"forged")
    with pytest.raises(ModelLoadError, match="declared hash"):
        admit_model_bundle(root)
