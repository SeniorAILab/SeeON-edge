from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import FramePacket, ModuleResult


@runtime_checkable
class Extractor(Protocol):
    def extract(self, packet: FramePacket) -> ModuleResult: ...


__all__ = ["Extractor"]
