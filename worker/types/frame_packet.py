from __future__ import annotations

from dataclasses import dataclass, field

from contracts.frame import Frame


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame: Frame = field(compare=False, hash=False)
    pts: float | None
    seq: int
    width: int
    height: int
    decode_time_ms: float


__all__ = ["FramePacket"]
