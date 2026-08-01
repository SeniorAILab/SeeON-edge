from __future__ import annotations

from typing import final


@final
class ModelLoadError(RuntimeError):
    __slots__: tuple[str, ...] = ()


@final
class ModelInputError(ValueError):
    __slots__: tuple[str, ...] = ()


@final
class FatalAcceleratorError(RuntimeError):
    """A GPU/CUDA error that requires retiring the context and exiting the process.

    Raised inside a CUDA adapter call when the GPU runtime, launch, sync,
    device-lost or unspecified failure occurs.  The process must write exactly
    one first-fault record and exit with code 4.  The context MUST NOT be
    reused or recreated in-process.
    """

    __slots__ = ("camera_id", "task")

    def __init__(
        self,
        message: str,
        *,
        camera_id: str = "",
        task: str = "",
    ) -> None:
        super().__init__(message)
        self.camera_id = camera_id
        self.task = task


__all__ = ["FatalAcceleratorError", "ModelInputError", "ModelLoadError"]
