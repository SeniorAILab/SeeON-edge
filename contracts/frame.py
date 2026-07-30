from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    time_sec: float
    image: NDArray[np.uint8]


@runtime_checkable
class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...
