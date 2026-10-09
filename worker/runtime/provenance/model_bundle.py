from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from contracts.model_selection import ContractError, ModelSelection, parse_model_selection

_BUNDLE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


class ModelBundleAdmissionError(RuntimeError):
    ...


@dataclass(frozen=True, slots=True)
class DesiredModelBundle:
    bundle_sha256: str
    selection: ModelSelection | None = None

    def __post_init__(self) -> None:
        if _BUNDLE_RE.fullmatch(self.bundle_sha256) is None:
            raise ModelBundleAdmissionError("desired bundle identity is invalid")


@dataclass(frozen=True, slots=True)
class ModelBundleProof:
    observed: Mapping[str, object]


def desired_model_bundle_from_selection_document(raw: object) -> DesiredModelBundle:
    try:
        selection = parse_model_selection(raw)
    except ContractError as exc:
        raise ModelBundleAdmissionError(str(exc)) from exc
    return DesiredModelBundle(selection.model_publication.bundle_sha256, selection)


def admit_model_bundle(models_root: Path, desired: DesiredModelBundle) -> ModelBundleProof:
    _require_directory(models_root, "models root")
    bundles_root = models_root / "bundles"
    _require_directory(bundles_root, "bundles root")
    root = bundles_root / desired.bundle_sha256
    _require_directory(root, "bundle root")
    _require_below(bundles_root, root)
    manifest_path = root / "manifest.json"
    raw = _read_regular(manifest_path, "manifest")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelBundleAdmissionError("bundle manifest is invalid JSON") from exc
    if not isinstance(document, dict) or _canonical_json(document).encode() + b"\n" != raw:
        raise ModelBundleAdmissionError("bundle manifest is not canonical")
    if document.get("schema_version") != 1:
        raise ModelBundleAdmissionError("bundle manifest schema mismatch")
    if document.get("bundle_sha256") != desired.bundle_sha256:
        raise ModelBundleAdmissionError("bundle identity mismatch")
    if (
        desired.selection is not None
        and document.get("runtime_format") != desired.selection.runtime_format
    ):
        raise ModelBundleAdmissionError("bundle runtime format mismatch")
    members = document.get("members")
    receipts = document.get("receipts", [])
    if (
        not isinstance(members, list)
        or not isinstance(receipts, list)
        or not isinstance(document.get("payload"), dict)
    ):
        raise ModelBundleAdmissionError("bundle manifest shape mismatch")
    _validate_bundle_identity(document, desired.bundle_sha256)
    observed_members = _verify_members(root, members)
    observed_receipts = _verify_members(root, receipts) if receipts else ()
    member_digests = {
        member_path: member["sha256"]
        for member_path, member in zip(observed_members, members, strict=True)
        if isinstance(member, dict)
    }
    _verify_exact_tree(root, {"manifest.json", *observed_members, *observed_receipts})
    observed = _freeze(
        {
            "bundle_sha256": desired.bundle_sha256,
            "members": tuple(observed_members),
            "member_digests": member_digests,
            "receipts": tuple(observed_receipts),
        }
    )
    return ModelBundleProof(observed=observed)


def _validate_bundle_identity(document: Mapping[str, object], expected: str) -> None:
    members = document["members"]
    payload = document["payload"]
    if not isinstance(members, list) or not isinstance(payload, dict):
        raise ModelBundleAdmissionError("bundle manifest shape mismatch")
    canonical_members: list[dict[str, object]] = []
    for member in members:
        if not isinstance(member, dict):
            raise ModelBundleAdmissionError("bundle member is invalid")
        try:
            canonical_members.append(
                {"path": member["path"], "sha256": member["sha256"], "size": member["size"]}
            )
        except KeyError as exc:
            raise ModelBundleAdmissionError("bundle member is invalid") from exc
    actual = hashlib.sha256(
        _canonical_json({"members": canonical_members, "payload": payload}).encode()
    ).hexdigest()
    if actual != expected:
        raise ModelBundleAdmissionError("bundle content identity mismatch")


def _verify_members(root: Path, members: list[object]) -> tuple[str, ...]:
    paths: list[str] = []
    for member in members:
        if not isinstance(member, dict):
            raise ModelBundleAdmissionError("bundle member is invalid")
        path = member.get("path")
        size = member.get("size")
        digest = member.get("sha256")
        if (
            not isinstance(path, str)
            or _RELATIVE_RE.fullmatch(path) is None
            or path == "manifest.json"
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _BUNDLE_RE.fullmatch(digest) is None
            or path in paths
        ):
            raise ModelBundleAdmissionError("bundle member is invalid")
        member_path = root / path
        _require_below(root, member_path)
        content = _read_regular(member_path, f"member {path}")
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ModelBundleAdmissionError(f"member mismatch: {path}")
        paths.append(path)
    if not paths:
        raise ModelBundleAdmissionError("bundle members missing")
    return tuple(paths)


def _verify_exact_tree(root: Path, expected: set[str]) -> None:
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    found_files: set[str] = set()
    found_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise ModelBundleAdmissionError(f"bundle contains unsafe path: {relative}")
        if stat.S_ISREG(info.st_mode):
            found_files.add(relative)
        else:
            found_directories.add(relative)
    if found_files != expected or found_directories != expected_directories:
        raise ModelBundleAdmissionError("bundle tree contains missing or extra filesystem nodes")


def _require_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ModelBundleAdmissionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ModelBundleAdmissionError(f"{label} is not a regular directory")


def _read_regular(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ModelBundleAdmissionError(f"{label} is not a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except ModelBundleAdmissionError:
        raise
    except OSError as exc:
        raise ModelBundleAdmissionError(f"{label} is unavailable") from exc


def _require_below(root: Path, path: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        if os.path.commonpath((str(resolved_root), str(path.resolve(strict=False)))) != str(
            resolved_root
        ):
            raise ModelBundleAdmissionError("bundle path escapes its root")
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
                raise ModelBundleAdmissionError("bundle contains symlink path")
    except OSError as exc:
        raise ModelBundleAdmissionError("bundle path is unavailable") from exc


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ModelBundleAdmissionError("bundle manifest contains non-JSON values") from exc


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(member) for key, member in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(member) for member in value)
    return value


__all__ = [
    "DesiredModelBundle",
    "ModelBundleAdmissionError",
    "ModelBundleProof",
    "admit_model_bundle",
    "desired_model_bundle_from_selection_document",
]
