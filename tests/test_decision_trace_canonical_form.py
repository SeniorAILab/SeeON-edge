import math

import pytest

from worker.types.trace import (
    TRACE_FLOAT_DECIMAL_PLACES,
    DecisionTraceSnapshot,
    canonical_trace_number,
    decision_trace_id,
)


def snapshot(values: dict[str, float]) -> DecisionTraceSnapshot:
    return DecisionTraceSnapshot(
        reason="fall-onset",
        previous_state="clear",
        current_state="fall",
        triggered=True,
        track_id=3,
        bed_id=None,
        values=values,
    )


def trace_id(values: dict[str, float]) -> str:
    return decision_trace_id(
        snapshot(values), module_qualified_id="fall.v2", effective_policy_id="policy-1"
    )


def test_negative_float_zero_is_normalized_to_positive_zero() -> None:
    normalized = canonical_trace_number(-0.0)

    assert normalized == 0.0
    assert math.copysign(1.0, normalized) == 1.0


def test_sub_micro_unit_noise_is_rounded_away() -> None:
    assert TRACE_FLOAT_DECIMAL_PLACES == 6
    assert canonical_trace_number(0.1234564999) == canonical_trace_number(0.123456)


def test_integers_stay_exact_integers() -> None:
    value = canonical_trace_number(2**62 + 1)

    assert type(value) is int
    assert value == 2**62 + 1


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, "1.0"])
def test_non_numeric_or_non_finite_trace_values_are_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_trace_number(value)


def test_trace_id_does_not_depend_on_value_insertion_order() -> None:
    forward = trace_id({"fall_probability": 0.9, "operating_threshold": 0.5})
    backward = trace_id({"operating_threshold": 0.5, "fall_probability": 0.9})

    assert forward == backward


def test_trace_id_is_stable_across_backend_float_noise() -> None:
    assert trace_id({"fall_probability": 0.9}) == trace_id({"fall_probability": 0.9000000001})


def test_trace_id_changes_with_the_policy_that_produced_the_decision() -> None:
    values = {"fall_probability": 0.9}
    other_policy = decision_trace_id(
        snapshot(values), module_qualified_id="fall.v2", effective_policy_id="policy-2"
    )

    assert trace_id(values) != other_policy
