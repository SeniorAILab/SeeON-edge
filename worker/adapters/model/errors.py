from __future__ import annotations

from typing import final

from shared.boundary import register_fatal


@final
class ModelLoadError(RuntimeError):
    __slots__: tuple[str, ...] = ()


@final
class ModelInputError(ValueError):
    __slots__: tuple[str, ...] = ()


@final
class FatalAcceleratorError(RuntimeError):
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


register_fatal(FatalAcceleratorError)

__all__ = ["FatalAcceleratorError", "ModelInputError", "ModelLoadError"]
