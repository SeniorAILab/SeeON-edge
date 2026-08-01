from __future__ import annotations

from typing import final


@final
class EncoderPolicyError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class EncoderStartError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class EncoderWriteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class ClipRemuxError(RuntimeError):
    def __init__(self, reason: str, returncode: int | None = None) -> None:
        self.reason = reason
        self.returncode = returncode
        super().__init__(reason)


@final
class CrossGenerationSegmentError(RuntimeError):
    def __init__(self, generations: tuple[int, ...]) -> None:
        self.generations = generations
        values = ", ".join(str(generation) for generation in generations)
        super().__init__(f"cannot finalize segments from multiple generations: {values}")


__all__ = [
    "ClipRemuxError",
    "CrossGenerationSegmentError",
    "EncoderPolicyError",
    "EncoderStartError",
    "EncoderWriteError",
]
