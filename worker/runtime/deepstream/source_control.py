"""Python-owned source generations and exact native-frame readiness."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, final

from worker.native.deepstream.control import ChildControlError
from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding


class SourceState(StrEnum):
    ABSENT = "absent"
    ADDING = "adding"
    STARTING = "starting"
    SOURCE_READY = "source_ready"
    REBUILDING = "rebuilding"
    DEGRADED = "degraded"
    REMOVING = "removing"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    camera_id: str
    state: SourceState
    source_generation: int | None
    stream_epoch: int | None


class SourceControl(Protocol):
    def add_source(self, camera_id: str, uri: str) -> SourceBinding: ...
    def remove_source(self, camera_id: str) -> None: ...
    def source_failure(self, camera_id: str, category: str) -> SourceBinding: ...


class FrameReceiver(Protocol):
    def pull_now(self, camera_id: str) -> MetadataFrame | None: ...
    def set_binding_handler(
        self,
        handler: Callable[[SourceBinding, MetadataFrame], None],
    ) -> None: ...


@final
class DarkSourceController:
    """Lifecycle registry where only an exact validated native frame establishes ready."""

    def __init__(
        self,
        control: SourceControl,
        slot: LatestMetadataSlot,
        receiver: FrameReceiver,
    ) -> None:
        self._control = control
        self._slot = slot
        self._receiver = receiver
        self._lock = threading.Lock()
        self._states: dict[str, SourceSnapshot] = {}
        self._receiver.set_binding_handler(self._on_epoch_frame)

    def snapshot(self, camera_id: str) -> SourceSnapshot:
        with self._lock:
            return self._states.get(
                camera_id,
                SourceSnapshot(camera_id, SourceState.ABSENT, None, None),
            )

    def _await_ready(self, binding: SourceBinding) -> SourceSnapshot:
        self._slot.register_source(binding)
        token = self._slot.subscribe(binding)
        _ = self._receiver.pull_now(binding.camera_id)
        _ = self._slot.wait_accepted(token, timeout_sec=2.0)
        ready = _snapshot(binding, SourceState.SOURCE_READY)
        self._set(ready)
        return ready

    def add(self, camera_id: str, uri: str) -> SourceSnapshot:
        current = self.snapshot(camera_id)
        self._set(
            SourceSnapshot(
                camera_id,
                SourceState.ADDING,
                current.source_generation,
                current.stream_epoch,
            )
        )
        try:
            binding = self._control.add_source(camera_id, uri)
        except ChildControlError:
            self._slot.remove_source(camera_id)
            self._set(
                SourceSnapshot(
                    camera_id,
                    SourceState.TOMBSTONED,
                    current.source_generation,
                    current.stream_epoch,
                )
            )
            raise
        starting = _snapshot(binding, SourceState.STARTING)
        self._set(starting)
        return self._await_ready(binding)

    def rebuild(self, camera_id: str, category: str) -> SourceSnapshot:
        current = self.snapshot(camera_id)
        self._set(
            SourceSnapshot(
                camera_id,
                SourceState.REBUILDING,
                current.source_generation,
                current.stream_epoch,
            )
        )
        try:
            binding = self._control.source_failure(camera_id, category)
        except ChildControlError:
            self._set(
                SourceSnapshot(
                    camera_id,
                    SourceState.DEGRADED,
                    current.source_generation,
                    current.stream_epoch,
                )
            )
            raise
        starting = _snapshot(binding, SourceState.STARTING)
        self._set(starting)
        return self._await_ready(binding)

    def remove(self, camera_id: str) -> SourceSnapshot:
        current = self.snapshot(camera_id)
        self._set(
            SourceSnapshot(
                camera_id,
                SourceState.REMOVING,
                current.source_generation,
                current.stream_epoch,
            )
        )
        self._control.remove_source(camera_id)
        self._slot.remove_source(camera_id)
        removed = SourceSnapshot(
            camera_id,
            SourceState.TOMBSTONED,
            current.source_generation,
            current.stream_epoch,
        )
        self._set(removed)
        return removed

    def _on_epoch_frame(self, binding: SourceBinding, metadata: MetadataFrame) -> None:
        if metadata.native_publish_sequence <= 0:
            return
        self._set(_snapshot(binding, SourceState.SOURCE_READY))

    def _set(self, snapshot: SourceSnapshot) -> None:
        with self._lock:
            self._states[snapshot.camera_id] = snapshot


def _snapshot(binding: SourceBinding, state: SourceState) -> SourceSnapshot:
    return SourceSnapshot(
        camera_id=binding.camera_id,
        state=state,
        source_generation=binding.source_generation,
        stream_epoch=binding.stream_epoch,
    )


__all__ = ["DarkSourceController", "SourceSnapshot", "SourceState"]
