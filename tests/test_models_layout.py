"""Validate the models/ layout + metadata.json contract.

models/ is gitignored, so git hooks cannot see it — this test is the
enforcement point for the ADR contract. Skips entirely when the tree is
absent (fresh clone / worktree without the symlink).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

ALLOWED_TOP_LEVEL = {"bed", "fall", "person", "pose"}
ALLOWED_SOURCES = {"downloaded", "trained", "third-party"}

pytestmark = pytest.mark.skipif(
    not any((MODELS_ROOT / "fall").glob("*/metadata.json")),
    reason=(
        "models/fall/ absent (fresh clone, unlinked worktree, or a "
        "persistent workspace where only models/pose/ has been populated "
        "by test_live_bench's YOLO auto-download, without any fall/ artifact)"
    ),
)


def _model_folders() -> list[Path]:
    """Every folder required to carry a metadata.json per the contract."""
    folders = [MODELS_ROOT / "pose"]
    fall = MODELS_ROOT / "fall"
    for child in sorted(fall.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "pretrained":
            folders.extend(d for d in sorted(child.iterdir()) if d.is_dir())
        else:
            folders.append(child)
    return folders


def test_top_level_is_function_axis_only() -> None:
    top = {d.name for d in MODELS_ROOT.iterdir() if d.is_dir()}
    unexpected = top - ALLOWED_TOP_LEVEL
    assert not unexpected, (
        f"models/ top level must be exactly {sorted(ALLOWED_TOP_LEVEL)}; "
        f"unexpected: {sorted(unexpected)} (origin/ephemeral axes belong in metadata.json)"
    )


def test_every_model_folder_has_metadata() -> None:
    missing = [str(f) for f in _model_folders() if not (f / "metadata.json").is_file()]
    assert not missing, f"metadata.json missing in: {missing}"


def test_metadata_required_fields() -> None:
    problems: list[str] = []
    for folder in _model_folders():
        path = folder / "metadata.json"
        if not path.is_file():
            continue  # covered by test_every_model_folder_has_metadata
        meta = json.loads(path.read_text())
        source = meta.get("source")
        if source not in ALLOWED_SOURCES:
            problems.append(f"{path}: source={source!r} not in {sorted(ALLOWED_SOURCES)}")
        if not meta.get("reacquire"):
            problems.append(f"{path}: reacquire missing or empty")
        if source == "trained" and not meta.get("version"):
            problems.append(f"{path}: trained artifact missing version")
    assert not problems, "metadata contract violations:\n" + "\n".join(problems)
