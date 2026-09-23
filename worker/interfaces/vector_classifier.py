"""Seam for a binary classifier over one fixed-length feature vector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorClassifierProtocol(Protocol):
    """Return the positive-class probability for one feature vector."""

    feature_dim: int

    def positive_probability(self, vector: Sequence[float]) -> float: ...


__all__ = ["VectorClassifierProtocol"]
