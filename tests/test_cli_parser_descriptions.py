from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[1]

PARSER_DESCRIPTIONS: Final = {
    "scripts/archive_preserved_sidecars.py": (
        "Durable, no-clobber archival transfer of owner-preserved working-tree files.\n"
        "\n"
        "Moves a fixed set of owner-owned files out of the repository into a pinned\n"
        "non-ephemeral archive root, then restores/removes the in-repo originals -- but\n"
        "only after the archive is provably durable, independent, and complete.\n"
        "\n"
        "Why this is not ``cp`` plus ``sha256sum``\n"
        "-----------------------------------------\n"
        "Comparing a fresh copy's digest to its source proves the bytes matched at one\n"
        "instant. It does not prove the copy is durable, independent, or unique:\n"
        "\n"
        "1. **Aliasing.** A destination that is a symlink (or hardlink) back into the\n"
        "   worktree digests identically to its source. Destroying the source then\n"
        "   leaves no independent copy. Closed by ``O_NOFOLLOW`` on create plus an\n"
        "   explicit ``(st_dev, st_ino)`` distinctness proof.\n"
        "2. **Page cache.** A just-written destination can be read back out of the page\n"
        "   cache and digest correctly while nothing has reached stable storage. A power\n"
        "   failure then loses it. Closed by ``fsync`` on the file *and* on the\n"
        "   containing directory (the directory entry created by the rename needs its\n"
        "   own flush).\n"
        "3. **Silent overwrite.** An unpinned or reused destination name can clobber an\n"
        "   earlier archive. POSIX ``rename(2)`` overwrites its target silently, so\n"
        "   publication uses ``link(2)`` -- which fails with ``EEXIST`` rather than\n"
        "   clobbering -- followed by unlinking the temporary name.\n"
        "\n"
        "Every source mutation is gated behind the whole batch succeeding. On any\n"
        "failure at any step the transaction halts having touched zero sources.\n"
    ),
    "scripts/ops/review-refused-evidence.py": (
        "Inspect and requeue evidence the backend refused.\n"
        "\n"
        "A 422 means the backend rejected a payload. The entry is retained rather than\n"
        "deleted, because deleting refused evidence and reporting it delivered is how 41\n"
        "real bed-exit events were destroyed on this deployment. Retention is only useful\n"
        "if an operator can act on it, and a retention area that fills has to be\n"
        "drainable -- otherwise the bound turns into a second stall.\n"
        "\n"
        "    # what is being held, and why\n"
        "    python scripts/ops/review-refused-evidence.py --state-dir /var/lib/seeon-state\n"
        "\n"
        "    # after the cause is fixed (a schema field, a relay version), put it back\n"
        "    python scripts/ops/review-refused-evidence.py --state-dir /var/lib/seeon-state "
        "--requeue\n"
        "\n"
        "Exit codes:\n"
        "  0  nothing retained, or the requeue completed\n"
        "  1  evidence is retained and needs review (inspection mode)\n"
        "  2  usage or environment error\n"
        "  3  requeue could not complete because the live queue is at capacity\n"
    ),
    "scripts/qa/batch_probe_compare.py": (
        "Compare two `scripts/qa/batch_probe.py` row sidecars frame by frame.\n"
        "\n"
        "Diagnostic only. Subject = the engine/batch under test, control = the batch-1\n"
        "engine on the same file sources (identical decoded frames, keyed by\n"
        "(pad, frame_number)). Reports the FP16 parity numbers the #503 acceptance\n"
        "uses: detection-count mismatches (with the scores of the odd box, which should\n"
        "sit at the pre-cluster gate), matched-box IoU, |dscore|, and keypoint L2 in\n"
        "network pixels. `--pad-permutation` remaps a run made with PAD_PERMUTATION so\n"
        "it can be compared against an unpermuted control.\n"
        "\n"
        "Usage:\n"
        "    python scripts/qa/batch_probe_compare.py --subject out.json.rows.jsonl         "
        "--control ctrl.json.rows.jsonl [--pad-permutation 12,3,7,0,9,1,11,5,2,10,4,8,6]         "
        "[--iou 0.9 --dscore 0.05 --kpt-px 2.0]\n"
    ),
    "scripts/qa/fall_model_recall_at_gate.py": (
        "Score a packaged fall model against recorded live-camera traces.\n"
        "\n"
        "Clean 300-frame training clips never exercise PTS resampling, track-id churn,\n"
        "or reconnect padding -- the exact continuity bugs this bundle exists to catch.\n"
        "This script instead replays recorded ``replay-trace-v2`` JSONL captures (real\n"
        "NvDCF track lifecycles, real gaps) through ``worker.replay.engine.replay()``,\n"
        "the same production compositor the worker boots, so the effective transition\n"
        "threshold (receipt vs. policy default) is resolved exactly as it is live.\n"
    ),
    "scripts/qa/golden_labeller_html.py": (
        "Render a golden worksheet CSV as a single offline HTML labelling page.\n"
        "\n"
        "The page plays each candidate clip from the local clip store, lets the owner\n"
        "pick ``real`` / ``false`` / ``unsure`` per episode, and exports the filled\n"
        "worksheet CSV from the browser (no server, no upload). The exported CSV is the\n"
        "input to ``tests_support/golden_episodes.py``.\n"
    ),
    "scripts/qa/trace_continuity.py": (
        "Per-track continuity statistics from a worker replay trace (`/traces/*.jsonl`).\n"
        "\n"
        "Diagnostic only. Answers \"are tracked people observed on consecutive frames?\"\n"
        "without touching production: presence ratio per NvDCF id, gap histogram between\n"
        "appearances, seq parity of object rows, rebirths of the same id, and the\n"
        "people-per-frame histogram. A healthy serving path shows gap-1 share >= 0.9 and\n"
        "no parity skew; the #503 defect showed gap-1 ~0.10, gap-2 ~0.42 and 90 % of\n"
        "object rows on one seq parity.\n"
        "\n"
        "Usage:\n"
        "    python scripts/qa/trace_continuity.py /tmp/p1b-flow/traces/<trace>.jsonl "
        "[--last-rows 27000]\n"
    ),
    "scripts/release_guard.py": (
        "Refuse to release unless every version carrier agrees with the tag.\n"
        "\n"
        "A release of this repository is cut by pushing an annotated tag shaped\n"
        "``seeon-edge-v<semver>``. That tag is the only thing an operator sees, so it\n"
        "must not be able to disagree with what the tree says about itself. This module\n"
        "is the guard: it reads the PRODUCT version out of every file that states one,\n"
        "requires them all to be identical, and — when a tag is being released —\n"
        "requires the tag to be exactly ``seeon-edge-v`` plus that version.\n"
        "\n"
        "It fails loudly: every carrier and its value is printed on the failure path, so\n"
        "the operator sees which file is out of step instead of a bare mismatch.\n"
        "\n"
        "Run it by hand before tagging:\n"
        "\n"
        "    python3 scripts/release_guard.py                       # lockstep only\n"
        "    python3 scripts/release_guard.py --tag seeon-edge-v0.1.0\n"
    ),
    "scripts/release_notes.py": (
        "Compose the release notes for a ``seeon-edge-v<semver>`` tag.\n"
        "\n"
        "The body has two parts:\n"
        "\n"
        "1. **Changes** — generated from the commit range since the previous\n"
        "   ``seeon-edge-v*`` tag. The very first release has no previous tag to diff\n"
        "   against, so it says so instead of dumping the whole history.\n"
        "2. **Images** — where the digest-pinned GHCR references live. They cannot be\n"
        "   inlined here: ``.github/workflows/edge-images.yml`` triggers on\n"
        "   ``release: published``, so the images are built *after* these notes exist.\n"
        "\n"
        "Both the rehearsal path and the real release path run this same code, so a\n"
        "rehearsal proves the notes compose before a tag is ever pushed.\n"
    ),
    "worker/tools/clip_playback_backfill.py": (
        "Create missing browser playback renditions for immutable evidence clips."
    ),
    "worker/tools/export_bed_seg_onnx.py": (
        "Export the bed YOLO26 segmentation weights to a digest-pinned ONNX artifact."
    ),
    "worker/tools/export_fall_onnx.py": (
        "Export a verified pose+bbox56 proxy bundle's Torch weights to ONNX at publish time."
    ),
    "worker/tools/export_pose_onnx.py": (
        "Export the nano pose weights to a dynamic-batch, digest-pinned ONNX artifact."
    ),
}


class ParserBuiltError(Exception):
    pass


@pytest.mark.parametrize(("relative_path", "expected"), sorted(PARSER_DESCRIPTIONS.items()))
def test_cli_parser_carries_its_description(
    monkeypatch: pytest.MonkeyPatch, relative_path: str, expected: str
) -> None:
    seen: list[str | None] = []

    def capture(parser: argparse.ArgumentParser, *_: object, **__: object) -> None:
        seen.append(parser.description)
        raise ParserBuiltError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", [relative_path])
    with pytest.raises(ParserBuiltError):
        runpy.run_path(str(ROOT / relative_path), run_name="__main__")
    assert expected
    assert seen == [expected]
