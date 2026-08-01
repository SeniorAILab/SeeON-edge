from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Final

from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import (
    Clock,
    ExtractorSpec,
    NamedExtractor,
    provision_extractors,
)

PRODUCTION_EXTRACTOR_NAMES: Final = ("pose", "person", "bed")


@dataclass(frozen=True, slots=True)
class SharedYoloExtractors:
    """Hold the process-shared YOLO extractors provisioned by runtime."""

    pose: NamedExtractor
    person: NamedExtractor
    bed: NamedExtractor

    @property
    def extractors(self) -> tuple[NamedExtractor, ...]:
        return (self.pose, self.person, self.bed)


def compose_yolo_extractors(
    serving_client: ServingClient,
    *,
    device: str,
    clock: Clock = perf_counter,
) -> SharedYoloExtractors:
    """Provision the production pose, person, and bed runners once."""
    specs = tuple(
        ExtractorSpec(
            module_name=name,
            task=name,
            options=(("device", device),),
        )
        for name in PRODUCTION_EXTRACTOR_NAMES
    )
    extractors = provision_extractors(serving_client, specs, clock=clock)
    return SharedYoloExtractors(
        pose=extractors[0],
        person=extractors[1],
        bed=extractors[2],
    )


__all__ = [
    "PRODUCTION_EXTRACTOR_NAMES",
    "SharedYoloExtractors",
    "compose_yolo_extractors",
]
