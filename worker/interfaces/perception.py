"""Capability protocols for worker-internal PerceptionFrame channels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from worker.types.perception_frame import (
    BedRegion,
    ChannelState,
    PerceptionFrameFailure,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
)


@runtime_checkable
class PersonBoxChannel(Protocol):
    state: ChannelState
    boxes: tuple[PersonBox, ...]


@runtime_checkable
class HumanPoseChannel(Protocol):
    state: ChannelState
    poses: tuple[tuple[tuple[int, int, float], ...], ...]


@runtime_checkable
class BedRegionChannel(Protocol):
    state: ChannelState
    regions: tuple[BedRegion, ...]


@runtime_checkable
class AssociationResult(Protocol):
    strategy: str
    track_ids: tuple[int, ...]
    selected_cue_indexes: tuple[int, ...]
    identity: PerceptionFrameIdentity
    cue_source: str


@runtime_checkable
class PerceptionFrameAdapter(Protocol):
    """Map inference outputs onto PerceptionFrameV1 or a typed failure."""

    def adapt(
        self,
        *args: object,
        **kwargs: object,
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def parse(
        self,
        payload: Mapping[str, object],
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def diagnostic(self, frame: PerceptionFrameV1) -> Mapping[str, object]: ...


__all__ = [
    "AssociationResult",
    "BedRegionChannel",
    "HumanPoseChannel",
    "PerceptionFrameAdapter",
    "PersonBoxChannel",
]
