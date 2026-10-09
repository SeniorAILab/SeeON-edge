from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worker.runtime.provenance.model_bundle import (
    ModelBundleAdmissionError,
    admit_model_bundle,
    desired_model_bundle_from_selection_document,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _selection(bundle_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 3,
        "model_publication": {
            "source_locator": "seeon/fall-model",
            "revision": "a" * 40,
            "bundle_sha256": bundle_sha256,
        },
        "runtime_format": "opaque-bundle-format",
        "transition_threshold": 0.5,
        "threshold_source": "default",
    }


def _bundle(
    tmp_path: Path,
    members: dict[str, bytes] | None = None,
    *,
    receipts: dict[str, bytes] | None = None,
) -> tuple[Path, object]:
    members = (
        {
            "model.onnx": b"model",
            "calibration.json": b'{"calibration": true}',
            "conformance/pose-bbox56-v1.json": b'{"conformance": true}',
            "bundle-manifest.json": b'{"schema_version":"bundle-manifest/proxy-v0"}',
        }
        if members is None
        else members
    )
    payload = {"identities": {"anything": "the producer chooses"}}
    member_records = [
        {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for path, content in members.items()
    ]
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            {"members": member_records, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    root = tmp_path / "models" / "bundles" / bundle_sha256
    root.mkdir(parents=True)
    receipt_records = []
    for path, content in {**members, **(receipts or {})}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    for path, content in (receipts or {}).items():
        receipt_records.append(
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        )
    manifest = {
        "schema_version": 1,
        "bundle_sha256": bundle_sha256,
        "runtime_format": "opaque-bundle-format",
        "members": member_records,
        "payload": payload,
    }
    if receipt_records:
        manifest["receipts"] = receipt_records
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return tmp_path / "models", desired_model_bundle_from_selection_document(
        _selection(bundle_sha256)
    )


def test_admission_returns_immutable_content_proof(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    proof = admit_model_bundle(models_root, desired)
    assert proof.observed["bundle_sha256"] == desired.bundle_sha256
    with pytest.raises(TypeError):
        proof.observed["bundle_sha256"] = "x"  # type: ignore[index]


def test_admission_rejects_extra_bundle_member(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "unexpected").write_text("unexpected")
    with pytest.raises(ModelBundleAdmissionError, match="bundle tree"):
        admit_model_bundle(models_root, desired)


def test_admission_uses_manifest_declared_member_set(tmp_path: Path) -> None:
    models_root, desired = _bundle(
        tmp_path,
        members={
            "runtime.bin": b"model",
            "contract.json": b'{"contract": true}',
            "calibration.json": b'{"calibration": true}',
            "conformance/pose-bbox56-v1.json": b'{"conformance": true}',
            "bundle-manifest.json": b'{"schema_version":"bundle-manifest/proxy-v0"}',
        },
    )
    assert admit_model_bundle(models_root, desired).observed["members"] == (
        "runtime.bin",
        "contract.json",
        "calibration.json",
        "conformance/pose-bbox56-v1.json",
        "bundle-manifest.json",
    )


def test_admission_requires_the_manifest_runtime_format(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["runtime_format"] = "different-format"
    manifest_path.write_bytes(_canonical(manifest))
    with pytest.raises(ModelBundleAdmissionError, match="runtime format"):
        admit_model_bundle(models_root, desired)




def test_admission_rejects_tampered_member_bytes(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "model.onnx").write_bytes(b"swapped")
    with pytest.raises(ModelBundleAdmissionError, match="member mismatch"):
        admit_model_bundle(models_root, desired)


def test_admission_accepts_receipt_files_without_reading_their_content(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path, receipts={"evaluation-receipt.json": b"not json"})
    proof = admit_model_bundle(models_root, desired)
    assert proof.observed["receipts"] == ("evaluation-receipt.json",)
