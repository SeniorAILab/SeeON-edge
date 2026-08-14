"""CLI for authoritative schema-9 clip consistency repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_authority import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLockedError


class _Arguments(argparse.Namespace):
    state_db: Path = Path()
    clip_store: Path = Path()
    apply: bool = False
    resume: bool = False
    maintenance_root: Path | None = None
    journal: Path | None = None
    quiescence_receipt: Path | None = None
    prebackup_receipt: Path | None = None
    state_uid: int = -1
    state_gid: int = -1
    state_db_mode: int = -1
    state_dir_mode: int = -1
    clip_uid: int = -1
    clip_gid: int = -1
    clip_dir_mode: int = -1
    tool_revision: str = ""
    ffprobe_bin: str = "ffprobe"


def _octal_mode(value: str) -> int:
    if len(value) != 4 or value[0] != "0" or any(digit not in "01234567" for digit in value):
        raise argparse.ArgumentTypeError("mode must be four octal digits, for example 0775")
    return int(value, 8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair schema-9 clip relations from authoritative final manifests."
    )
    _ = parser.add_argument("--state-db", type=Path, required=True)
    _ = parser.add_argument("--clip-store", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    _ = mode.add_argument("--apply", action="store_true")
    _ = mode.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--maintenance-root", type=Path)
    _ = parser.add_argument("--journal", type=Path)
    _ = parser.add_argument("--quiescence-receipt", type=Path)
    _ = parser.add_argument("--prebackup-receipt", type=Path)
    _ = parser.add_argument("--state-uid", type=int, required=True)
    _ = parser.add_argument("--state-gid", type=int, required=True)
    _ = parser.add_argument("--state-db-mode", type=_octal_mode, required=True)
    _ = parser.add_argument("--state-dir-mode", type=_octal_mode, required=True)
    _ = parser.add_argument("--clip-uid", type=int, required=True)
    _ = parser.add_argument("--clip-gid", type=int, required=True)
    _ = parser.add_argument("--clip-dir-mode", type=_octal_mode, required=True)
    _ = parser.add_argument("--tool-revision", required=True)
    _ = parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv, namespace=_Arguments())
    try:
        receipt = repair_clip_consistency(
            RepairRequest(
                state_db=arguments.state_db,
                clip_store=arguments.clip_store,
                authority=RepairAuthority(
                    state_uid=arguments.state_uid,
                    state_gid=arguments.state_gid,
                    state_db_mode=arguments.state_db_mode,
                    state_dir_mode=arguments.state_dir_mode,
                    clip_uid=arguments.clip_uid,
                    clip_gid=arguments.clip_gid,
                    clip_dir_mode=arguments.clip_dir_mode,
                    tool_revision=arguments.tool_revision,
                ),
                apply=arguments.apply,
                resume=arguments.resume,
                maintenance_root=arguments.maintenance_root,
                journal_path=arguments.journal,
                quiescence_receipt=arguments.quiescence_receipt,
                prebackup_receipt=arguments.prebackup_receipt,
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
