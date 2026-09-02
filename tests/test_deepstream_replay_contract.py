"""D-6 replay corpus and PerceptionFrame timeline comparison contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_COMPARATOR = _ROOT / "scripts/qa/compare_perception_timeline.py"
_CORPUS = _ROOT / "scripts/qa/deepstream-canary/make-replay-corpus.sh"


def _frame(box_source: str = "pose") -> dict[str, object]:
    identity = {
        "worker_boot_id": "boot-a",
        "camera_id": "camera-1",
        "stream_epoch": 3,
        "seq": 8,
        "source_pts": 90000,
    }
    return {
        "version": "PerceptionFrameV1",
        "box_source": box_source,
        "identity": identity,
        "person_box": {
            "state": "inferred",
            "boxes": [{"x1": 1, "y1": 2, "x2": 30, "y2": 40, "confidence": 0.9}],
        },
        "human_pose": {"state": "inferred", "poses": [[{"x": 2, "y": 3, "score": 0.8}]]},
        "bed_region": {
            "state": "inferred",
            "regions": [
                {
                    "x1": 4,
                    "y1": 5,
                    "x2": 50,
                    "y2": 60,
                    "confidence": 0.7,
                    "polygon": [[4, 5], [50, 5]],
                }
            ],
        },
        "association": {
            "strategy": "legacy-greedy-bbox-iou.v1",
            "track_ids": [11],
            "selected_cue_indexes": [0],
            "cue_source": "person_box",
            "identity": deepcopy(identity),
        },
    }


def _compare(
    tmp_path: Path,
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    source: str = "pose",
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    left, right, output = (
        tmp_path / "baseline.jsonl",
        tmp_path / "candidate.jsonl",
        tmp_path / "result.json",
    )
    left.write_text("".join(json.dumps(frame) + "\n" for frame in baseline), encoding="utf-8")
    right.write_text("".join(json.dumps(frame) + "\n" for frame in candidate), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(_COMPARATOR),
            str(left),
            str(right),
            "--box-source",
            source,
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return process, json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["pose", "person"])
def test_comparator_passes_each_declared_box_source(tmp_path: Path, source: str) -> None:
    baseline = [_frame(source)]
    candidate = deepcopy(baseline)
    candidate[0]["identity"]["worker_boot_id"] = "other-boot"  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate, source)
    assert process.returncode == 0
    assert report == {
        "baseline_frames": 1,
        "candidate_frames": 1,
        "first_mismatch": None,
        "result": "PASS",
    }


def test_comparator_confidence_tolerance_boundary_and_overage(tmp_path: Path) -> None:
    baseline, candidate = [_frame()], [_frame()]
    candidate[0]["person_box"]["boxes"][0]["confidence"] = 0.900006  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate)
    assert process.returncode == 0 and report["result"] == "PASS"
    candidate[0]["person_box"]["boxes"][0]["confidence"] = 0.9000061  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate)
    assert (
        process.returncode == 1
        and report["first_mismatch"] == "$[0].person_box.boxes[0].confidence"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("person_box", "boxes", 0, "x2"), 31),
        (("person_box", "boxes"), []),
        (("human_pose", "state"), "skipped"),
        (("association", "selected_cue_indexes"), [1]),
        (("bed_region", "regions", 0, "x1"), 9),
    ],
)
def test_comparator_rejects_exact_contract_changes(
    tmp_path: Path, path: tuple[object, ...], value: object
) -> None:
    baseline, candidate = [_frame()], [_frame()]
    target: object = candidate[0]
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate)
    assert process.returncode == 1
    assert report["result"] == "FAIL"


def test_comparator_rejects_frame_count_and_durable_identity_changes(tmp_path: Path) -> None:
    process, report = _compare(tmp_path, [_frame()], [])
    assert process.returncode == 1 and report["first_mismatch"] == "$: frame count"
    baseline, candidate = [_frame()], [_frame()]
    candidate[0]["identity"]["source_pts"] = 90001  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate)
    assert process.returncode == 1 and report["first_mismatch"] == "$[0].identity.source_pts"


@pytest.mark.parametrize("forbidden_key", ["raw_pixels", "input_tensor"])
def test_comparator_rejects_raw_media_payloads(tmp_path: Path, forbidden_key: str) -> None:
    baseline, candidate = [_frame()], [_frame()]
    candidate[0][forbidden_key] = [0]  # type: ignore[index]
    process, report = _compare(tmp_path, baseline, candidate)
    assert process.returncode == 1
    assert forbidden_key in str(report["first_mismatch"])


def test_corpus_script_is_remux_only_and_refuses_existing_output(tmp_path: Path) -> None:
    script = _CORPUS.read_text(encoding="utf-8")
    assert "-c:v copy" in script
    assert "-an" in script
    assert "ffprobe" in script
    assert "http" not in script.lower()
    source = tmp_path / "not-a-video.mp4"
    source.write_bytes(b"not a video")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "replay-v1.mp4").write_bytes(b"published")
    process = subprocess.run(
        ["bash", str(_CORPUS), str(source), str(root)], text=True, capture_output=True, check=False
    )
    assert process.returncode == 1
    assert (root / "replay-v1.mp4").read_bytes() == b"published"


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe required"
)
def test_corpus_script_remuxes_h264_hashes_and_publishes_read_only(tmp_path: Path) -> None:
    source, root = tmp_path / "source.mp4", tmp_path / "corpus"
    created = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if created.returncode:
        pytest.skip("ffmpeg lacks libx264")
    process = subprocess.run(
        ["bash", str(_CORPUS), str(source), str(root)], text=True, capture_output=True, check=False
    )
    assert process.returncode == 0, process.stderr
    output, sidecar = root / "replay-v1.mp4", root / "replay-v1.mp4.sha256"
    codec = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert codec == "h264"
    assert (
        sidecar.read_text(encoding="utf-8")
        == f"{hashlib.sha256(output.read_bytes()).hexdigest()}  replay-v1.mp4\n"
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    refused = subprocess.run(
        ["bash", str(_CORPUS), str(source), str(root)], text=True, capture_output=True, check=False
    )
    assert refused.returncode == 1
