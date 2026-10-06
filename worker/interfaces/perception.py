from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from contracts.runner import BedRunnerResult, PersonRunnerResult, PoseRunnerResult
from worker.types.perception_frame import (
    LEGACY_ASSOCIATION_STRATEGY,
    PERSON_BOX_CUE_SOURCE,
    PerceptionFrameFailure,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
)


@runtime_checkable
class PerceptionFrameAdapter(Protocol):
    def adapt(
        self,
        *,
        identity: PerceptionFrameIdentity,
        pose: PoseRunnerResult | None = None,
        person: PersonRunnerResult | None = None,
        bed: BedRunnerResult | None = None,
        track_ids: tuple[int, ...] | None = None,
        selected_cue_indexes: tuple[int, ...] | None = None,
        association_identity: PerceptionFrameIdentity | None = None,
        association_strategy: str = LEGACY_ASSOCIATION_STRATEGY,
        association_cue_source: str = PERSON_BOX_CUE_SOURCE,
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def parse(
        self,
        payload: Mapping[str, object],
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def diagnostic(self, frame: PerceptionFrameV1) -> Mapping[str, object]: ...


__all__ = [
    "PerceptionFrameAdapter",
]
