from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from worker.types.metadata import MetadataFrame, SourceBinding


class MetadataSlot(Protocol):
    def register_source(self, binding: SourceBinding) -> object: ...

    def remove_source(self, camera_id: str) -> None: ...

    def publish(self, metadata: MetadataFrame) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    session_id: int
    camera_id: str
    path: str
    duration_ms: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SourceStatus:
    camera_id: str
    binding: SourceBinding
    live: bool
    reconnects: int


@dataclass(frozen=True, slots=True)
class MediaPlaneStatus:
    sources: tuple[SourceStatus, ...]
    engine_identity: str
    nvenc_sessions_active: int
    fatal_error: str | None = None


@runtime_checkable
class MediaPlane(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> MediaPlaneStatus: ...

    def add_source(self, camera_id: str, uri: str) -> SourceBinding: ...

    def remove_source(self, camera_id: str) -> None: ...

    def source_failure(self, camera_id: str, category: str) -> SourceBinding: ...

    def snapshot(self, camera_id: str) -> bytes:
        ...

    def native_snapshot(self, camera_id: str) -> bytes:
        ...

    def start_recording(
        self,
        camera_id: str,
        *,
        lookback_sec: int,
        duration_sec: int,
        on_sealed: Callable[[RecordingInfo], None],
    ) -> int:
        ...

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        ...


class RecordingRefused(RuntimeError):
    ...


class EarlyStopUnsupported(RuntimeError):
    ...


class SourceRosterFixed(RuntimeError):
    ...


class SnapshotUnavailable(RuntimeError):
    ...


class OnDemandSnapshotUnsupported(SnapshotUnavailable):
    ...
