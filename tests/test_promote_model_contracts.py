from __future__ import annotations

import pytest

from contracts.model_selection import (
    SELECTION_KEYS,
    ContractError,
    canonical_digest,
    canonical_json_bytes,
    parse_model_selection,
)
from worker.runtime.provenance.model_bundle import desired_model_bundle_from_selection_document

BUNDLE_SHA256 = "b" * 64


def desired_raw() -> dict[str, object]:
    return {
        "schema_version": 3,
        "model_publication": {
            "source_locator": "seeon/fall-model",
            "revision": "0" * 40,
            "bundle_sha256": BUNDLE_SHA256,
        },
        "runtime_format": "opaque-bundle-format",
        "transition_threshold": 0.5,
        "threshold_source": "default",
    }


def test_canonical_identity_is_key_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})


def test_selection_round_trips_and_keys_match_the_exposed_set() -> None:
    assert frozenset(desired_raw()) == SELECTION_KEYS
    assert parse_model_selection(desired_raw()).as_dict() == desired_raw()


def test_runtime_admission_selection_parser_keeps_the_bundle_pin() -> None:
    desired = desired_model_bundle_from_selection_document(desired_raw())
    assert desired.bundle_sha256 == BUNDLE_SHA256
    assert desired.selection is not None
    assert desired.selection.runtime_format == "opaque-bundle-format"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model_publication", "revision"), "main"),
        (("model_publication", "bundle_sha256"), "sha256:" + BUNDLE_SHA256),
        (("runtime_format",), ""),
        (("schema_version",), 2),
        (("threshold_source",), "green"),
    ],
)
def test_desired_rejects_nonimmutable_or_invalid_contract_values(
    path: tuple[str, ...], value: object
) -> None:
    raw = desired_raw()
    target: dict[str, object] = raw
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    with pytest.raises(ContractError):
        parse_model_selection(raw)


@pytest.mark.parametrize(
    "key",
    [
        "evaluation_receipt_digest",
        "field_evaluation_receipt_digest",
        "calibration_digest",
        "dataset_publication",
        "model_family",
        "worker_image_digest",
    ],
)
def test_contract_rejects_removed_ceremony_and_pin_fields(key: str) -> None:
    raw = desired_raw()
    raw[key] = "a" * 64
    with pytest.raises(ContractError):
        parse_model_selection(raw)
