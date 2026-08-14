"""Explicit split filesystem authority for clip consistency maintenance."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_authority_types import (
    RepairAuthority,
)
from worker.pipeline.output.evidence.clip_consistency_io import (
    reject_lexical_parent_components,
    validate_directory,
    validate_no_symlink_components,
    validate_regular,
)
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError

AUTHORITY_KEYS = frozenset(RepairAuthority.__dataclass_fields__)


def validate_authority(authority: RepairAuthority) -> None:
    identifiers = (
        authority.state_uid,
        authority.state_gid,
        authority.clip_uid,
        authority.clip_gid,
    )
    modes = (
        authority.state_db_mode,
        authority.state_dir_mode,
        authority.clip_dir_mode,
    )
    if any(identifier < 0 for identifier in identifiers):
        raise ClipConsistencyError("authority_invalid", "authority IDs must be nonnegative")
    if any(mode < 0 or mode > 0o7777 for mode in modes):
        raise ClipConsistencyError("authority_invalid", "authority mode is invalid")
    if authority.state_db_mode & 0o022:
        raise ClipConsistencyError("authority_invalid", "state database mode is writable")
    if authority.state_dir_mode & 0o022:
        raise ClipConsistencyError("authority_invalid", "state directory mode is writable")
    if authority.clip_dir_mode & 0o002:
        raise ClipConsistencyError("authority_invalid", "clip directory is world writable")
    revision = authority.tool_revision
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ClipConsistencyError("authority_invalid", "tool revision must be a full git SHA")
    packaged = os.environ.get("CLIP_CONSISTENCY_TOOL_REVISION")
    if packaged is not None and revision != packaged:
        raise ClipConsistencyError("authority_invalid", "tool revision differs from image")


def validate_state_authority(state_db: Path, authority: RepairAuthority) -> None:
    validate_authority(authority)
    validate_authority_ancestors(state_db.parent, authority)
    validate_directory(
        state_db.parent,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        owner_controlled=False,
        exact_mode=authority.state_dir_mode,
        label="state database directory",
    )
    validate_regular(
        state_db,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        exact_mode=authority.state_db_mode,
        label="state database",
    )
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{state_db}{suffix}")
        if sidecar.exists() or sidecar.is_symlink():
            validate_regular(
                sidecar,
                expected_uid=authority.state_uid,
                expected_gid=authority.state_gid,
                exact_mode=None,
                label=f"state database {suffix[1:].upper()}",
            )


def validate_maintenance_authority(path: Path, authority: RepairAuthority) -> None:
    validate_authority_ancestors(path.parent, authority)
    validate_directory(
        path,
        expected_uid=authority.state_uid,
        expected_gid=authority.state_gid,
        owner_controlled=True,
        label="maintenance root",
    )


def validate_clip_authority(clip_store: Path, authority: RepairAuthority) -> None:
    validate_authority(authority)
    validate_authority_ancestors(clip_store.parent, authority)
    clips_root = clip_store / "clips"
    staging_root = clips_root / ".staging"
    for path, label in (
        (clip_store, "clip store"),
        (clips_root, "clips root"),
        (staging_root, "staging root"),
    ):
        validate_directory(
            path,
            expected_uid=authority.clip_uid,
            expected_gid=authority.clip_gid,
            owner_controlled=False,
            exact_mode=authority.clip_dir_mode,
            label=label,
        )
    for entry in clip_store.iterdir():
        if entry == clips_root:
            continue
        info = entry.lstat()
        lock_owners = {
            (authority.state_uid, authority.state_gid),
            (authority.clip_uid, authority.clip_gid),
        }
        if (
            entry.name != ".worker.lock"
            or not stat.S_ISREG(info.st_mode)
            or (info.st_uid, info.st_gid) not in lock_owners
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ClipConsistencyError("unsafe_path", "clip-store root entry is invalid")
    for root, directories, files in os.walk(clips_root, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in (*directories, *files):
            entry = root_path / name
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ClipConsistencyError("unsafe_path", "clip tree contains a symlink")
            if info.st_uid != authority.clip_uid or info.st_gid != authority.clip_gid:
                raise ClipConsistencyError("unsafe_path", "clip tree owner differs")
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                if mode & 0o002:
                    raise ClipConsistencyError("unsafe_path", "clip directory is world writable")
            elif stat.S_ISREG(info.st_mode):
                if mode & 0o002:
                    raise ClipConsistencyError("unsafe_path", "clip file is world writable")
            else:
                raise ClipConsistencyError("unsafe_path", "clip tree entry type is invalid")


def validate_authority_ancestors(path: Path, authority: RepairAuthority) -> None:
    reject_lexical_parent_components(path)
    validate_no_symlink_components(path)
    absolute = path.absolute()
    for ancestor in reversed((absolute, *absolute.parents)):
        info = ancestor.stat(follow_symlinks=False)
        owner = (info.st_uid, info.st_gid)
        if owner not in {(0, 0), (authority.state_uid, authority.state_gid)}:
            raise ClipConsistencyError("unsafe_path", "path ancestor owner differs")
        mode = stat.S_IMODE(info.st_mode)
        writable = mode & 0o022
        sticky_root = owner == (0, 0) and bool(mode & stat.S_ISVTX)
        if writable and not sticky_root:
            raise ClipConsistencyError("unsafe_path", "path ancestor is writable")


__all__ = [
    "AUTHORITY_KEYS",
    "RepairAuthority",
    "validate_authority",
    "validate_clip_authority",
    "validate_maintenance_authority",
    "validate_state_authority",
]
