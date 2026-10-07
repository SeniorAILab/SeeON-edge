from __future__ import annotations

from typing import TypeAlias

FallModelInput: TypeAlias = tuple[float, ...] | tuple[tuple[float, ...], ...]

__all__ = ["FallModelInput"]
