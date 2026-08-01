"""Camera-keyed latest JPEG storage and live-view publication."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import cv2

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output.overlay import OverlayEncodingError, OverlayRenderer
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class LatestFrame:
    jpeg: bytes
    seq: int
    frame_index: int
    content_type: str = "image/jpeg"


class LatestFrameStore:
    """Retain one non-consuming latest JPEG value for every known camera."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frames: dict[str, LatestFrame] = {}
        self._known_camera_ids: set[str] = set()

    def register_camera(self, camera_id: str) -> None:
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._condition.notify_all()

    def publish_jpeg(
        self,
        camera_id: str,
        jpeg: bytes,
        *,
        frame_index: int,
        seq: int | None = None,
    ) -> None:
        latest = LatestFrame(
            jpeg=bytes(jpeg),
            seq=frame_index if seq is None else seq,
            frame_index=frame_index,
        )
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._frames[camera_id] = latest
            self._condition.notify_all()

    def get_latest(self, camera_id: str) -> LatestFrame | None:
        with self._condition:
            return self._frames.get(camera_id)

    def is_known(self, camera_id: str) -> bool:
        with self._condition:
            return camera_id in self._known_camera_ids

    def wait_for_latest(
        self,
        camera_id: str,
        *,
        previous: LatestFrame | None,
        timeout: float,
    ) -> LatestFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frames.get(camera_id) is not previous,
                timeout=timeout,
            )
            return self._frames.get(camera_id)


class LiveViewRenderer(Protocol):
    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...],
    ) -> bytes: ...


class LiveViewSubscriber:
    """Render optional live output without changing inference state or flow."""

    def __init__(
        self,
        store: LatestFrameStore,
        renderer: LiveViewRenderer | None = None,
    ) -> None:
        self._store = store
        self._renderer = renderer if renderer is not None else OverlayRenderer()

    def publish(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bool:
        try:
            jpeg = self._renderer.encode_jpeg(packet, observation, debug_snapshots)
        except (cv2.error, OverlayEncodingError):
            return False
        self._store.publish_jpeg(
            packet.camera_id,
            jpeg,
            seq=packet.seq,
            frame_index=packet.frame.index,
        )
        return True


__all__ = [
    "LatestFrame",
    "LatestFrameStore",
    "LiveViewSubscriber",
]
