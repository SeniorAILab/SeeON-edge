"""Fall-model ports shared by domain classifiers and concrete adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from worker.types import FallModelInput

FallState = Literal["background", "fall_transition", "fallen"]


@dataclass(frozen=True, slots=True)
class FallOutputContract:
    """Versioned meaning of one model output tensor and its policy projection."""

    contract_id: str
    version: int
    supported_states: tuple[FallState, ...]
    model_output_size: int
    model_class_order: tuple[str, ...]
    activation: Literal["sigmoid", "softmax"]

    @property
    def unsupported_states(self) -> tuple[FallState, ...]:
        return tuple(
            state
            for state in ("background", "fall_transition", "fallen")
            if state not in self.supported_states
        )

    def manifest_value(self) -> dict[str, object]:
        return {
            "id": self.contract_id,
            "version": self.version,
            "supported_states": list(self.supported_states),
        }


BINARY_FALL_OUTPUT_CONTRACT = FallOutputContract(
    contract_id="fall.binary-transition-sigmoid",
    version=1,
    supported_states=("background", "fall_transition"),
    model_output_size=1,
    model_class_order=("non_fall", "fall_transition_proxy"),
    activation="sigmoid",
)
THREE_STATE_FALL_OUTPUT_CONTRACT = FallOutputContract(
    contract_id="fall.three-state-softmax",
    version=1,
    supported_states=("background", "fall_transition", "fallen"),
    model_output_size=3,
    model_class_order=("background", "fall_transition", "fallen"),
    activation="softmax",
)


@dataclass(frozen=True, slots=True)
class BinaryFallScoreEvidence:
    """Observed binary score and calibration used to produce one result.

    ``class_origins`` follows the probability field order below: background is
    the complement of the calibrated score, fall_transition is the
    temperature-scaled sigmoid, and fallen is a synthetic constant zero.
    """

    raw_logit: float
    applied_temperature: float
    output_contract: FallOutputContract = field(
        default=BINARY_FALL_OUTPUT_CONTRACT,
        init=False,
    )
    class_origins: tuple[
        Literal["derived_complement"],
        Literal["temperature_sigmoid"],
        Literal["constant_zero"],
    ] = field(
        default=("derived_complement", "temperature_sigmoid", "constant_zero"),
        init=False,
    )


@dataclass(frozen=True, slots=True)
class ThreeStateFallScoreEvidence:
    """Observed three-state logits and calibration used to produce one result."""

    raw_logits: tuple[float, float, float]
    applied_temperature: float
    output_contract: FallOutputContract = field(
        default=THREE_STATE_FALL_OUTPUT_CONTRACT,
        init=False,
    )
    class_origins: tuple[
        Literal["temperature_softmax"],
        Literal["temperature_softmax"],
        Literal["temperature_softmax"],
    ] = field(
        default=("temperature_softmax", "temperature_softmax", "temperature_softmax"),
        init=False,
    )


@dataclass(frozen=True, slots=True)
class FallProbabilities:
    """Policy probabilities with optional evidence from a declared output contract.

    ``fall_transition`` is already calibrated. ``model_evidence=None`` means
    that no source score was observed; it does not imply any output contract,
    a unit temperature, or a reconstructed logit.
    """

    background: float
    fall_transition: float
    fallen: float
    model_evidence: BinaryFallScoreEvidence | ThreeStateFallScoreEvidence | None = field(
        default=None,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        for value in (self.background, self.fall_transition, self.fallen):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fall probabilities must be finite values in [0, 1]")


@runtime_checkable
class FallModelProtocol(Protocol):
    """Models score one ``(30, 56)`` pose+bbox56 window on the CPU."""

    def predict(self, features: FallModelInput) -> FallProbabilities: ...


__all__ = [
    "BINARY_FALL_OUTPUT_CONTRACT",
    "BinaryFallScoreEvidence",
    "FallOutputContract",
    "FallModelProtocol",
    "FallProbabilities",
    "FallState",
    "THREE_STATE_FALL_OUTPUT_CONTRACT",
    "ThreeStateFallScoreEvidence",
]
