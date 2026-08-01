from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CpuAvConfig:
    camera_id: str
    url: str
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000


__all__ = ["CpuAvConfig"]
