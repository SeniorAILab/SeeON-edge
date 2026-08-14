"""Descriptor-backed identities revalidated at clip-repair phase boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from worker.pipeline.output.evidence.clip_consistency_authority_types import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    device: int
    inode: int
    file_type: int
    uid: int
    gid: int
    mode: int
    content_sha256: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "file_type": self.file_type,
            "uid": self.uid,
            "gid": self.gid,
            "mode": self.mode,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> FileIdentity:
        if not isinstance(value, dict):
            raise ClipConsistencyError("journal_invalid", "file identity is invalid")
        payload = cast("dict[str, object]", value)
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ClipConsistencyError("journal_invalid", "file identity keys differ")
        integers = ("device", "inode", "file_type", "uid", "gid", "mode")
        if not isinstance(payload.get("path"), str) or any(
            not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool)
            for key in integers
        ):
            raise ClipConsistencyError("journal_invalid", "file identity type differs")
        digest = payload.get("content_sha256")
        if digest is not None and (not isinstance(digest, str) or not _is_sha256(digest)):
            raise ClipConsistencyError("journal_invalid", "file identity digest differs")
        return cls(
            path=cast(str, payload["path"]),
            device=cast(int, payload["device"]),
            inode=cast(int, payload["inode"]),
            file_type=cast(int, payload["file_type"]),
            uid=cast(int, payload["uid"]),
            gid=cast(int, payload["gid"]),
            mode=cast(int, payload["mode"]),
            content_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    state_db: FileIdentity
    ancestors: tuple[FileIdentity, ...]
    maintenance: tuple[FileIdentity, ...]
    clip_entries: tuple[FileIdentity, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state_db": self.state_db.to_dict(),
            "ancestors": [item.to_dict() for item in self.ancestors],
            "maintenance": [item.to_dict() for item in self.maintenance],
            "clip_entries": [item.to_dict() for item in self.clip_entries],
        }

    @classmethod
    def from_dict(cls, value: object) -> AuthoritySnapshot:
        if not isinstance(value, dict):
            raise ClipConsistencyError("journal_invalid", "authority snapshot is invalid")
        payload = cast("dict[str, object]", value)
        if set(payload) != {"state_db", "ancestors", "maintenance", "clip_entries"}:
            raise ClipConsistencyError("journal_invalid", "authority snapshot keys differ")
        return cls(
            state_db=FileIdentity.from_dict(payload["state_db"]),
            ancestors=_identity_list(payload["ancestors"]),
            maintenance=_identity_list(payload["maintenance"]),
            clip_entries=_identity_list(payload["clip_entries"]),
        )


@dataclass(frozen=True, slots=True)
class ProofIdentity:
    file: FileIdentity
    operation_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file.to_dict(),
            "operation_digest": self.operation_digest,
        }

    def semantic_dict(self) -> dict[str, int | str | None]:
        return self.file.to_dict()

    @classmethod
    def from_dict(cls, value: object) -> ProofIdentity:
        if not isinstance(value, dict):
            raise ClipConsistencyError("journal_invalid", "proof identity is invalid")
        payload = cast("dict[str, object]", value)
        if set(payload) != {"file", "operation_digest"}:
            raise ClipConsistencyError("journal_invalid", "proof identity keys differ")
        operation_digest = payload.get("operation_digest")
        if not isinstance(operation_digest, str) or not _is_sha256(operation_digest):
            raise ClipConsistencyError("journal_invalid", "proof operation digest differs")
        return cls(FileIdentity.from_dict(payload["file"]), operation_digest)


def capture_proof_identity(path: Path, authority: RepairAuthority) -> ProofIdentity:
    descriptor = _open_absolute(path)
    try:
        info = os.fstat(descriptor)
        raw = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            raw += chunk
        payload = _proof_payload(raw)
        operation_digest = cast(str, payload["operation_digest"])
        semantic = dict(payload)
        semantic.pop("operation_digest")
        digest = hashlib.sha256(
            json.dumps(
                semantic, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        identity = _identity(str(path.absolute()), info, digest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClipConsistencyError("authority_drift", "proof identity is invalid") from exc
    finally:
        os.close(descriptor)
    _require_regular(identity, authority.state_uid, authority.state_gid, 0o600, "proof")
    return ProofIdentity(identity, operation_digest)


def capture_authority_snapshot(
    *,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    tracked_maintenance: tuple[Path, ...],
    authority: RepairAuthority,
) -> AuthoritySnapshot:
    state_identity = identity_for_path(state_db)
    _require_regular(
        state_identity,
        authority.state_uid,
        authority.state_gid,
        authority.state_db_mode,
        "state database",
    )
    ancestor_paths = _canonical_ancestors(
        (state_db.parent, clip_store.parent, maintenance_root.parent)
    )
    ancestors = tuple(identity_for_path(path) for path in ancestor_paths)
    maintenance_paths = tuple(dict.fromkeys((maintenance_root, *tracked_maintenance)))
    maintenance = tuple(
        identity_for_path(path, hash_content=path.suffix == ".sqlite3")
        for path in maintenance_paths
    )
    for identity in maintenance:
        if identity.uid != authority.state_uid or identity.gid != authority.state_gid:
            raise ClipConsistencyError("authority_drift", "maintenance owner differs")
        if identity.path == str(maintenance_root.absolute()):
            if identity.file_type != stat.S_IFDIR or identity.mode & 0o077:
                raise ClipConsistencyError("authority_drift", "maintenance root differs")
        elif identity.file_type != stat.S_IFREG or identity.mode != 0o600:
            raise ClipConsistencyError("authority_drift", "maintenance artifact differs")
    clip_entries = _scan_clip_tree(clip_store, authority)
    return AuthoritySnapshot(state_identity, ancestors, maintenance, clip_entries)


def validate_phase_authority(
    snapshot: AuthoritySnapshot,
    *,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    tracked_maintenance: tuple[Path, ...],
    authority: RepairAuthority,
    quarantine: tuple[tuple[str, str], ...],
    quarantine_state: str,
    deleted_quarantine: tuple[str, ...] = (),
) -> None:
    current = capture_authority_snapshot(
        state_db=state_db,
        clip_store=clip_store,
        maintenance_root=maintenance_root,
        tracked_maintenance=tracked_maintenance,
        authority=authority,
    )
    if current.state_db != snapshot.state_db:
        raise ClipConsistencyError("authority_drift", "state database identity changed")
    if current.ancestors != snapshot.ancestors:
        raise ClipConsistencyError("authority_drift", "path ancestor identity changed")
    if current.maintenance != snapshot.maintenance:
        raise ClipConsistencyError("authority_drift", "maintenance artifact identity changed")
    expected = _expected_clip_entries(
        snapshot.clip_entries,
        current.clip_entries,
        quarantine,
        quarantine_state,
        frozenset(deleted_quarantine),
    )
    if current.clip_entries != expected:
        raise ClipConsistencyError("authority_drift", "clip tree identity changed")


def validate_journal_identity(
    path: Path,
    authority: RepairAuthority,
    *,
    expected_sha256: str | None = None,
    expected_identity: FileIdentity | None = None,
) -> FileIdentity:
    identity = identity_for_path(path, hash_content=True)
    _require_regular(identity, authority.state_uid, authority.state_gid, 0o600, "journal")
    if expected_sha256 is not None and identity.content_sha256 != expected_sha256:
        raise ClipConsistencyError("authority_drift", "journal content changed")
    if expected_identity is not None and identity != expected_identity:
        raise ClipConsistencyError("authority_drift", "journal identity changed")
    return identity


def identity_for_path(path: Path, *, hash_content: bool = False) -> FileIdentity:
    descriptor = _open_absolute(path)
    try:
        info = os.fstat(descriptor)
        digest = (
            _sha256_descriptor(descriptor)
            if hash_content and stat.S_ISREG(info.st_mode)
            else None
        )
        return _identity(str(path.absolute()), info, digest)
    finally:
        os.close(descriptor)


def _scan_clip_tree(clip_store: Path, authority: RepairAuthority) -> tuple[FileIdentity, ...]:
    descriptor = _open_absolute(clip_store)
    try:
        root_info = os.fstat(descriptor)
        root = _identity(".", root_info, None)
        _require_clip_entry(root, authority, root=True)
        entries = [root]
        _scan_directory(descriptor, PurePosixPath(), entries, authority)
        return tuple(sorted(entries, key=lambda item: item.path))
    finally:
        os.close(descriptor)


def _scan_directory(
    descriptor: int,
    relative: PurePosixPath,
    entries: list[FileIdentity],
    authority: RepairAuthority,
) -> None:
    for name in sorted(os.listdir(descriptor)):
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        try:
            child = os.open(name, flags, dir_fd=descriptor)
        except OSError as exc:
            raise ClipConsistencyError("authority_drift", "clip entry could not be opened") from exc
        try:
            info = os.fstat(child)
            child_relative = relative / name
            logical = child_relative.as_posix()
            digest = (
                _sha256_descriptor(child)
                if stat.S_ISREG(info.st_mode) and name == "manifest.json"
                else None
            )
            identity = _identity(logical, info, digest)
            _require_clip_entry(identity, authority, root=False)
            entries.append(identity)
            if stat.S_ISDIR(info.st_mode):
                _scan_directory(child, child_relative, entries, authority)
        finally:
            os.close(child)


def _expected_clip_entries(
    baseline: tuple[FileIdentity, ...],
    current: tuple[FileIdentity, ...],
    quarantine: tuple[tuple[str, str], ...],
    state: str,
    deleted: frozenset[str],
) -> tuple[FileIdentity, ...]:
    current_paths = {item.path for item in current}
    result: list[FileIdentity] = []
    for identity in baseline:
        replacement = identity.path
        omitted = False
        for original, held in quarantine:
            if identity.path == original or identity.path.startswith(f"{original}/"):
                if held in deleted:
                    omitted = True
                    break
                suffix = identity.path[len(original) :]
                if state == "original":
                    replacement = identity.path
                elif state == "held":
                    replacement = f"{held}{suffix}"
                elif state == "either":
                    original_present = original in current_paths
                    held_present = held in current_paths
                    if original_present == held_present:
                        raise ClipConsistencyError(
                            "authority_drift", "quarantine location is ambiguous"
                        )
                    replacement = f"{held}{suffix}" if held_present else identity.path
                else:
                    raise ClipConsistencyError("authority_drift", "phase state is invalid")
                break
        if not omitted:
            result.append(
                FileIdentity(
                    path=replacement,
                    device=identity.device,
                    inode=identity.inode,
                    file_type=identity.file_type,
                    uid=identity.uid,
                    gid=identity.gid,
                    mode=identity.mode,
                    content_sha256=identity.content_sha256,
                )
            )
    return tuple(sorted(result, key=lambda item: item.path))


def _open_absolute(path: Path) -> int:
    absolute = path.absolute()
    parts = absolute.parts[1:]
    descriptor = os.open(absolute.anchor or "/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
            if index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    else:
        return descriptor


def _proof_payload(raw: bytes) -> dict[str, object]:
    loaded: object = json.loads(raw.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("proof is not an object")
    payload = cast("dict[str, object]", loaded)
    operation_digest = payload.get("operation_digest")
    if not isinstance(operation_digest, str) or not _is_sha256(operation_digest):
        raise ValueError("proof operation digest is invalid")
    return payload


def _identity(path: str, info: os.stat_result, digest: str | None) -> FileIdentity:
    return FileIdentity(
        path=path,
        device=info.st_dev,
        inode=info.st_ino,
        file_type=stat.S_IFMT(info.st_mode),
        uid=info.st_uid,
        gid=info.st_gid,
        mode=stat.S_IMODE(info.st_mode),
        content_sha256=digest,
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, position, os.SEEK_SET)
    return digest.hexdigest()


def _require_regular(
    identity: FileIdentity,
    uid: int,
    gid: int,
    mode: int,
    label: str,
) -> None:
    if (
        identity.file_type != stat.S_IFREG
        or identity.uid != uid
        or identity.gid != gid
        or identity.mode != mode
    ):
        raise ClipConsistencyError("authority_drift", f"{label} authority differs")


def _require_clip_entry(
    identity: FileIdentity, authority: RepairAuthority, *, root: bool
) -> None:
    if root and identity.file_type != stat.S_IFDIR:
        raise ClipConsistencyError("authority_drift", "clip root type differs")
    if identity.path == ".worker.lock":
        allowed = {
            (authority.state_uid, authority.state_gid),
            (authority.clip_uid, authority.clip_gid),
        }
        if (
            identity.file_type != stat.S_IFREG
            or (identity.uid, identity.gid) not in allowed
            or identity.mode != 0o600
        ):
            raise ClipConsistencyError("authority_drift", "clip lock authority differs")
        return
    if identity.uid != authority.clip_uid or identity.gid != authority.clip_gid:
        raise ClipConsistencyError("authority_drift", "clip entry owner differs")
    if identity.file_type not in {stat.S_IFDIR, stat.S_IFREG} or identity.mode & 0o002:
        raise ClipConsistencyError("authority_drift", "clip entry type or mode differs")
    if identity.path in {".", "clips", "clips/.staging"} and (
        identity.file_type != stat.S_IFDIR or identity.mode != authority.clip_dir_mode
    ):
        raise ClipConsistencyError("authority_drift", "clip root mode differs")


def _canonical_ancestors(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    values: dict[str, Path] = {}
    for path in paths:
        absolute = path.absolute()
        for candidate in (absolute, *absolute.parents):
            values[str(candidate)] = candidate
    return tuple(values[key] for key in sorted(values))


def _identity_list(value: object) -> tuple[FileIdentity, ...]:
    if not isinstance(value, list):
        raise ClipConsistencyError("journal_invalid", "identity list is invalid")
    return tuple(FileIdentity.from_dict(item) for item in cast("list[object]", value))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "AuthoritySnapshot",
    "FileIdentity",
    "ProofIdentity",
    "capture_authority_snapshot",
    "capture_proof_identity",
    "identity_for_path",
    "validate_journal_identity",
    "validate_phase_authority",
]
