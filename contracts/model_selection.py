from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final = 3
POSE_BBOX56_PREPROCESSING_IDENTITY: Final = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_LOCATOR_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class ContractError(ValueError): ...


@dataclass(frozen=True)
class ModelPublication:
    source_locator: str
    revision: str
    bundle_sha256: str


@dataclass(frozen=True)
class ModelSelection:
    model_publication: ModelPublication
    runtime_format: str
    transition_threshold: float
    threshold_source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_publication": {
                "source_locator": self.model_publication.source_locator,
                "revision": self.model_publication.revision,
                "bundle_sha256": self.model_publication.bundle_sha256,
            },
            "runtime_format": self.runtime_format,
            "transition_threshold": self.transition_threshold,
            "threshold_source": self.threshold_source,
        }


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("ascii")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(raw: object, where: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{where} must be an object")
    return raw


def _exact_keys(raw: Mapping[str, object], expected: frozenset[str], where: str) -> None:
    actual = frozenset(raw)
    if actual != expected:
        raise ContractError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _string(raw: Mapping[str, object], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}.{key} must be a non-empty string")
    return value


def _digest(raw: Mapping[str, object], key: str, where: str) -> str:
    value = _string(raw, key, where)
    if _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{where}.{key} must be a lowercase 64-hex SHA-256 digest")
    return value


def _publication(raw: object, where: str) -> ModelPublication:
    value = _object(raw, where)
    _exact_keys(value, frozenset({"source_locator", "revision", "bundle_sha256"}), where)
    source_locator = _string(value, "source_locator", where)
    if _SOURCE_LOCATOR_RE.fullmatch(source_locator) is None:
        raise ContractError(f"{where}.source_locator must be an owner/repository name")
    revision = _string(value, "revision", where)
    if _REVISION_RE.fullmatch(revision) is None:
        raise ContractError(f"{where}.revision must be a lowercase 40-hex immutable ref")
    return ModelPublication(source_locator, revision, _digest(value, "bundle_sha256", where))


def _probability(raw: Mapping[str, object], key: str, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a probability")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ContractError(f"{where}.{key} must be in [0, 1]")
    return parsed


def _threshold_source(raw: Mapping[str, object], key: str, where: str) -> str:
    value = _string(raw, key, where)
    if value not in {"default", "receipt"}:
        raise ContractError(f"{where}.{key} must be default or receipt")
    return value


SELECTION_KEYS: Final = frozenset(
    {
        "schema_version",
        "model_publication",
        "runtime_format",
        "transition_threshold",
        "threshold_source",
    }
)


def parse_model_selection(raw: object) -> ModelSelection:
    where = "model-selection"
    value = _object(raw, where)
    _exact_keys(value, SELECTION_KEYS, where)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{where}.schema_version must be {SCHEMA_VERSION}")
    return ModelSelection(
        model_publication=_publication(
            value.get("model_publication"), f"{where}.model_publication"
        ),
        runtime_format=_string(value, "runtime_format", where),
        transition_threshold=_probability(value, "transition_threshold", where),
        threshold_source=_threshold_source(value, "threshold_source", where),
    )


__all__ = [
    "POSE_BBOX56_PREPROCESSING_IDENTITY",
    "SCHEMA_VERSION",
    "SELECTION_KEYS",
    "ContractError",
    "ModelPublication",
    "ModelSelection",
    "canonical_digest",
    "canonical_json_bytes",
    "parse_model_selection",
]
