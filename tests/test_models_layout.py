"""Validate the models/ layout + metadata.json contract.

models/ is gitignored, so git hooks cannot see it — this test is the
enforcement point for the ADR contract. The metadata.json folder checks skip
when that tree is absent (fresh clone / worktree without the symlink). The
fetch-script and Dockerfile characterization below always run.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
FETCH_MODELS = REPO_ROOT / "scripts" / "fetch-models.sh"

ALLOWED_TOP_LEVEL = {"bed", "fall", "person", "pose"}
ALLOWED_SOURCES = {"downloaded", "trained", "third-party"}
TRACKED_LSTM_SIDECARS = (
    Path("models/fall/lstm/arch.json"),
    Path("models/fall/lstm/metadata.yaml"),
)
_LEGACY_METADATA_REASON = (
    "models/fall/ absent (fresh clone, unlinked worktree, or a "
    "persistent workspace where only models/pose/ has been populated "
    "by test_live_bench's YOLO auto-download, without any fall/ artifact)"
)


def _has_legacy_fall_metadata_json() -> bool:
    return any((MODELS_ROOT / "fall").glob("*/metadata.json"))


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


def _git_ls_files(*paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *paths],
        check=True,
        capture_output=True,
    )
    return [item.decode() for item in result.stdout.split(b"\0") if item]


def _write_stub(path: Path, name: str, body: str) -> Path:
    command = path / name
    command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    return command


def test_git_tracks_only_packaged_lstm_sidecars() -> None:
    tracked = {Path(path) for path in _git_ls_files("models")}
    assert tracked == set(TRACKED_LSTM_SIDECARS)


def test_fetch_models_script_pins_public_revision_and_local_dest() -> None:
    text = FETCH_MODELS.read_text(encoding="utf-8")

    assert 'HF_REPO="Berom0227/eldercare-fall-models"' in text
    assert 'HF_REVISION="d67887844bfd2e4b1ca3f3275f770b0b05e23aba"' in text
    assert "ML_WORKER_FETCH_MODELS_DEST" in text
    assert 'fetch_one "model.pt"' in text
    assert 'fetch_one "metadata.json" "metadata.upstream.json"' in text
    assert "metadata.yaml" in text
    assert "arch.json" in text


def test_fetch_models_unknown_argument_exits_usage() -> None:
    result = subprocess.run(
        ["bash", str(FETCH_MODELS), "--unexpected"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_fetch_models_skips_existing_destination_without_network(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "lstm"
    dest.mkdir()
    (dest / "model.pt").write_bytes(b"synthetic-lstm-weights")
    (dest / "metadata.upstream.json").write_text("{}", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir, "curl", 'echo "curl must not run for an idempotent skip" >&2; exit 1')

    result = subprocess.run(
        ["bash", str(FETCH_MODELS)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ML_WORKER_FETCH_MODELS_DEST": str(dest),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "already exists; skipping" in result.stdout
    assert "curl must not run" not in result.stderr
    assert (dest / "model.pt").read_bytes() == b"synthetic-lstm-weights"


def test_dockerfiles_do_not_fetch_or_copy_model_weights() -> None:
    for name in ("Dockerfile.edge", "Dockerfile.backend"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "fetch-models" not in text
        assert "huggingface.co" not in text
        assert "COPY models" not in text
        assert "ADD models" not in text
        assert ".pt" not in text
        assert "mkdir -p /app/models/" in text


@pytest.mark.skipif(not _has_legacy_fall_metadata_json(), reason=_LEGACY_METADATA_REASON)
def test_top_level_is_function_axis_only() -> None:
    top = {d.name for d in MODELS_ROOT.iterdir() if d.is_dir()}
    unexpected = top - ALLOWED_TOP_LEVEL
    assert not unexpected, (
        f"models/ top level must be exactly {sorted(ALLOWED_TOP_LEVEL)}; "
        f"unexpected: {sorted(unexpected)} (origin/ephemeral axes belong in metadata.json)"
    )


@pytest.mark.skipif(not _has_legacy_fall_metadata_json(), reason=_LEGACY_METADATA_REASON)
def test_every_model_folder_has_metadata() -> None:
    missing = [str(f) for f in _model_folders() if not (f / "metadata.json").is_file()]
    assert not missing, f"metadata.json missing in: {missing}"


@pytest.mark.skipif(not _has_legacy_fall_metadata_json(), reason=_LEGACY_METADATA_REASON)
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
