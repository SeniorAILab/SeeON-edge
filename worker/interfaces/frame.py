from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.frame import Frame
from worker.types import ConverterCapabilities, FrameLease


@runtime_checkable
class FrameMaterializer(Protocol):
    name: str
    capabilities: ConverterCapabilities

    def materialize(self, lease: FrameLease) -> FrameLease: ...


@runtime_checkable
class HostFrameView(Protocol):
    name: str

    def view(self, lease: FrameLease) -> Frame: ...


__all__ = ["FrameMaterializer", "HostFrameView"]
