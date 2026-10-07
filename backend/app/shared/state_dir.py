from __future__ import annotations

from pathlib import Path


def resolve_state_dir(runtime: str = "ml-api") -> Path:
    return Path.home() / ".local" / "state" / runtime


__all__ = ["resolve_state_dir"]
