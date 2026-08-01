from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from worker.types import BusinessEvent, FramePacket

_CameraT = TypeVar("_CameraT", contravariant=True)
_ProfileT = TypeVar("_ProfileT", contravariant=True)
_GeometryT = TypeVar("_GeometryT", contravariant=True)
_SegmentT = TypeVar("_SegmentT", contravariant=True)
_ArtifactT = TypeVar("_ArtifactT", covariant=True)


@runtime_checkable
class EncoderSession(Protocol):
    """One long-lived derivative encoder session owned by a camera."""

    def write(self, packet: FramePacket) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ClipEncoder(Protocol[_CameraT, _ProfileT, _GeometryT]):
    """Open a camera encoder without resolving profile policy internally."""

    def open(
        self,
        camera: _CameraT,
        profile: _ProfileT,
        geometry: _GeometryT,
    ) -> EncoderSession: ...


@runtime_checkable
class ClipFinalizer(Protocol[_SegmentT, _ArtifactT]):
    """Finalize selected completed segments into an immutable clip artifact."""

    def finalize(
        self,
        segments: Sequence[_SegmentT],
        event: BusinessEvent,
    ) -> _ArtifactT: ...


__all__ = ["ClipEncoder", "ClipFinalizer", "EncoderSession"]
