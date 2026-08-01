"""Worker-internal envelope types (image-free, frozen dataclasses)."""

from __future__ import annotations

from worker.types.business_event import BusinessEvent
from worker.types.decision_input import DecisionInput
from worker.types.frame_packet import FramePacket
from worker.types.module_result import ModuleResult

__all__ = [
    "BusinessEvent",
    "DecisionInput",
    "FramePacket",
    "ModuleResult",
]
