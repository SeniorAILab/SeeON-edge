"""CLI for authoritative schema-9 clip consistency repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_repair import (
    repair_clip_consistency,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLockedError


class _Arguments(argparse.Namespace):
    store_dir: Path = Path()
    apply: bool = False
    prebackup_receipt: Path | None = None
    backup_dir: Path | None = None
    receipt_dir: Path | None = None
    ffprobe_bin: str = "ffprobe"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair schema-9 clip relations from authoritative final manifests."
    )
    _ = parser.add_argument("store_dir", type=Path)
    _ = parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the reported repair; default is read-only dry-run",
    )
    _ = parser.add_argument("--prebackup-receipt", type=Path)
    _ = parser.add_argument("--backup-dir", type=Path)
    _ = parser.add_argument("--receipt-dir", type=Path)
    _ = parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv, namespace=_Arguments())
    try:
        receipt = repair_clip_consistency(
            RepairRequest(
                store_dir=arguments.store_dir,
                apply=arguments.apply,
                prebackup_receipt=arguments.prebackup_receipt,
                backup_dir=arguments.backup_dir,
                receipt_dir=arguments.receipt_dir,
                ffprobe_bin=arguments.ffprobe_bin,
            )
        )
    except (ClipConsistencyError, ClipStoreLockedError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
