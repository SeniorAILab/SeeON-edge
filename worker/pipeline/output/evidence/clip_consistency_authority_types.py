"""Value object for explicit clip maintenance filesystem authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RepairAuthority:
    state_uid: int
    state_gid: int
    state_db_mode: int
    state_dir_mode: int
    clip_uid: int
    clip_gid: int
    clip_dir_mode: int
    tool_revision: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()


__all__ = ["RepairAuthority"]
