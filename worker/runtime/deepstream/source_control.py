"""Python-owned source generations and internal dark-child readiness."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from worker.native.deepstream.control import DeepStreamControlClient
from worker.native.deepstream.metadata import LatestMetadataSlot, MetadataReceiver, SourceBinding


class SourceState(StrEnum):
    ABSENT = "absent"
    ADDING = "adding"
    STARTING = "starting"
    SOURCE_READY = "source_ready"
    REBUILDING = "rebuilding"
    REMOVING = "removing"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    camera_id: str
    state: SourceState
    source_generation: int | None
    stream_epoch: int | None


@final
class DarkSourceController:
    """Mutating lifecycle registry; only validated metadata establishes ready."""

    def __init__(
        self,
        control: DeepStreamControlClient,
        slot: LatestMetadataSlot,
        receiver: MetadataReceiver,
    ) -> None:
        self._control = control
        self._slot = slot
        self._receiver = receiver
        self._lock = threading.Lock()
        self._states: dict[str, SourceSnapshot] = {}

    def snapshot(self, camera_id: str) -> SourceSnapshot:
        with self._lock:
            return self._states.get(
                camera_id,
                SourceSnapshot(camera_id, SourceState.ABSENT, None, None),
            )

    def add(self, camera_id: str, uri: str) -> SourceSnapshot:
        self._set(SourceSnapshot(camera_id, SourceState.ADDING, None, None))
        binding = self._control.add_source(camera_id, uri)
        self._slot.register_source(binding)
        starting = _snapshot(binding, SourceState.STARTING)
        self._set(starting)
        subscription = self._receiver.subscription()
        self._control.emit_metadata(camera_id)
        self._receiver.wait_received(subscription, timeout_sec=2.0)
        metadata = self._slot.peek(camera_id)
        if metadata is None:
            return starting
        ready = _snapshot(binding, SourceState.SOURCE_READY)
        self._set(ready)
        return ready

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
        binding = self._control.source_failure(camera_id, category)
        self._slot.register_source(binding)
        subscription = self._receiver.subscription()
        self._control.emit_metadata(camera_id)
        self._receiver.wait_received(subscription, timeout_sec=2.0)
        ready = _snapshot(binding, SourceState.SOURCE_READY)
        self._set(ready)
        return ready

    def pause_metadata(self) -> None:
        self._receiver.pause()

    def resume_metadata(self) -> None:
        subscription = self._receiver.subscription()
        self._receiver.resume()
        self._receiver.wait_received(subscription, timeout_sec=2.0)

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
