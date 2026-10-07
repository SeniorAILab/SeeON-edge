from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "worker/tools/fetch_models/manifest.json"


def _artifacts() -> dict[str, dict[str, object]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(entry["path"]): entry for entry in manifest["artifacts"]}


def test_every_onnx_artifact_lists_its_digest_sidecar() -> None:
    artifacts = _artifacts()
    onnx_paths = [
        path
        for path in artifacts
        if path.endswith(".onnx") and path.split("/")[0] in {"pose", "bed"}
    ]
    assert onnx_paths, "manifest lists no ONNX artifacts"
    missing = [path for path in onnx_paths if f"{path}.sha256" not in artifacts]
    assert missing == [], f"ONNX artifacts without a fetched digest sidecar: {missing}"


def test_sidecar_entries_are_the_65_byte_digest_line_of_their_model() -> None:
    artifacts = _artifacts()
    for path, entry in artifacts.items():
        if not path.endswith(".onnx.sha256"):
            continue
        model = artifacts[path.removesuffix(".sha256")]
        expected_line = f"{model['sha256']}\n".encode("ascii")
        assert entry["size"] == len(expected_line) == 65
        assert entry["sha256"] == hashlib.sha256(expected_line).hexdigest()
