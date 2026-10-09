from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from worker.adapters.model.errors import ModelLoadError

_SHA256: Final = r"^[0-9a-f]{64}$"
_RELATIVE: Final = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
_MANIFEST: Final = "manifest.json"


class _Member(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    path: Annotated[str, Field(pattern=_RELATIVE)]
    sha256: Annotated[str, Field(pattern=_SHA256)]
    size: Annotated[int, Field(ge=0)]


class _Manifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)

    schema_version: Literal[1]
    runtime_format: str
    members: Annotated[list[_Member], Field(min_length=1)]
    receipts: list[_Member] = []


def admit_model_bundle(root: Path) -> Mapping[str, str]:
    try:
        raw = (root / _MANIFEST).read_bytes()
        document = json.loads(raw)
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        if canonical.encode() + b"\n" != raw:
            raise ModelLoadError("manifest.json is not canonical JSON")
        manifest = _Manifest.model_validate(document)
        if manifest.runtime_format != "onnxruntime":
            raise ModelLoadError(
                f"runtime_format {manifest.runtime_format!r} is not onnxruntime; "
                "the flow profile cannot run it"
            )
        declared = [*manifest.members, *manifest.receipts]
        _require_exact_tree(root, [member.path for member in declared])
        for member in declared:
            content = (root / member.path).read_bytes()
            if len(content) != member.size or hashlib.sha256(content).hexdigest() != member.sha256:
                raise ModelLoadError(f"member {member.path} does not match its declared hash")
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"unreadable or malformed bundle: {exc}") from exc
    return MappingProxyType({member.path: member.sha256 for member in declared})


def _require_exact_tree(root: Path, members: list[str]) -> None:
    expected = {_MANIFEST, *members}
    if len(expected) != len(members) + 1:
        raise ModelLoadError("manifest.json declares a duplicate member or lists itself")
    found: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            found.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(info.st_mode):
            raise ModelLoadError(f"{path.relative_to(root)} is a symlink or special file")
    if found != expected:
        raise ModelLoadError(
            f"file tree differs from manifest.json: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )


__all__ = ["admit_model_bundle"]
