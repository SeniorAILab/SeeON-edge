from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from worker.types import FallModelInput


@dataclass(frozen=True, slots=True)
class BinaryFallScoreEvidence:
    raw_logit: float
    applied_temperature: float
    class_origins: tuple[
        Literal["derived_complement"],
        Literal["temperature_sigmoid"],
        Literal["constant_zero"],
    ] = field(
        default=("derived_complement", "temperature_sigmoid", "constant_zero"),
        init=False,
    )


@dataclass(frozen=True, slots=True)
class FallProbabilities:
    background: float
    fall_transition: float
    fallen: float
    model_evidence: BinaryFallScoreEvidence | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for value in (self.background, self.fall_transition, self.fallen):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fall probabilities must be finite values in [0, 1]")


@runtime_checkable
class FallModelProtocol(Protocol):
    def predict(self, features: FallModelInput) -> FallProbabilities: ...


__all__ = [
    "BinaryFallScoreEvidence",
    "FallModelProtocol",
    "FallProbabilities",
]
