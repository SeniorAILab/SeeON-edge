"""Strict parsing for versioned fall-model output contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from contracts.model_selection import canonical_digest
from worker.adapters.model.errors import ModelLoadError
from worker.interfaces.fall_model import (
    BINARY_FALL_OUTPUT_CONTRACT,
    THREE_STATE_FALL_OUTPUT_CONTRACT,
    FallOutputContract,
)

_CONTRACTS: Final = {
    (
        BINARY_FALL_OUTPUT_CONTRACT.contract_id,
        BINARY_FALL_OUTPUT_CONTRACT.version,
    ): BINARY_FALL_OUTPUT_CONTRACT,
    (
        THREE_STATE_FALL_OUTPUT_CONTRACT.contract_id,
        THREE_STATE_FALL_OUTPUT_CONTRACT.version,
    ): THREE_STATE_FALL_OUTPUT_CONTRACT,
}
_KEYS: Final = frozenset({"id", "version", "supported_states"})


def parse_fall_output_contract(
    bundle_manifest: object,
) -> tuple[FallOutputContract, str]:
    """Return a known exact contract and the digest bound by model selection."""
    if not isinstance(bundle_manifest, Mapping):
        raise ModelLoadError("invalid bundle-manifest.json")
    raw = bundle_manifest.get("output_contract")
    if not isinstance(raw, Mapping):
        raise ModelLoadError("bundle manifest must declare output_contract")
    if frozenset(raw) != _KEYS:
        raise ModelLoadError(
            "output_contract must contain exactly id, version, and supported_states"
        )
    contract_id = raw.get("id")
    version = raw.get("version")
    if (
        not isinstance(contract_id, str)
        or not contract_id
        or not isinstance(version, int)
        or isinstance(version, bool)
    ):
        raise ModelLoadError("output_contract id and version are invalid")
    contract = _CONTRACTS.get((contract_id, version))
    if contract is None:
        raise ModelLoadError(
            f"unsupported output_contract id/version: {contract_id!r} version {version!r}"
        )
    expected = contract.manifest_value()
    if dict(raw) != expected:
        raise ModelLoadError(
            f"output_contract metadata mismatch: observed {dict(raw)!r}, expected {expected!r}"
        )
    return contract, canonical_digest(expected)


def parse_packaged_fall_output_contract(
    bundle_manifest: object,
) -> tuple[FallOutputContract, str]:
    """Resolve a packaged bundle's contract, admitting the pre-contract v0 shape.

    A bundle published before ``output_contract`` existed declares its class
    order at the manifest's top level instead. When ``output_contract`` is
    absent and that legacy ``class_order`` matches the binary contract's
    exactly, the bundle is binary. Every other bundle -- with an
    ``output_contract``, or without one and a non-matching legacy
    ``class_order`` -- goes through the strict parser, so a bundle that omits
    both still fails closed. Only the packaged loaders call this; the
    admitted path always requires an explicit ``output_contract``.
    """
    if isinstance(bundle_manifest, Mapping) and "output_contract" not in bundle_manifest:
        legacy_class_order = bundle_manifest.get("class_order")
        if legacy_class_order == list(BINARY_FALL_OUTPUT_CONTRACT.model_class_order):
            expected = BINARY_FALL_OUTPUT_CONTRACT.manifest_value()
            return BINARY_FALL_OUTPUT_CONTRACT, canonical_digest(expected)
    return parse_fall_output_contract(bundle_manifest)


__all__ = ["parse_fall_output_contract", "parse_packaged_fall_output_contract"]
