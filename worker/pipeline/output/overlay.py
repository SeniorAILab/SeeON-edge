"""Best-effort overlays rendered from frozen frame-analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output._overlay_primitives import (
    BED_COLOR,
    BED_EXIT_STATUS_COLOR,
    BED_PRESENT_STATUS_COLOR,
    BED_ROI_TEXT_COLOR,
    PERSON_COLOR,
    POSE_DOT_COLOR,
    draw_box,
    draw_caption,
    draw_label,
    draw_pose,
    draw_region,
)
from worker.types import FramePacket

MAX_SNAPSHOT_BYTES: Final = 200 * 1024


class OverlayEncodingError(RuntimeError):
    """Raised when OpenCV cannot encode a rendered overlay."""


@dataclass(frozen=True, slots=True)
class OverlayRenderer:
    """Render frozen inference outputs without mutating the shared frame."""

    show_pose: bool = False

    def render(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> NDArray[np.uint8]:
        image = packet.frame.image.copy()
        for box in observation.boxes:
            draw_box(image, box, PERSON_COLOR)
            draw_label(image, "person", box.x1, max(12, box.y1 - 4), PERSON_COLOR)
        for box in observation.bed_boxes:
            draw_region(image, box, BED_COLOR, fill=True)
            draw_label(image, "bed", box.x1, max(12, box.y1 - 4), BED_COLOR)
        for snapshot in debug_snapshots:
            _draw_bed_exit_debug(image, snapshot)
        if self.show_pose:
            for keypoints in observation.keypoints:
                draw_pose(
                    image,
                    keypoints,
                    color=POSE_DOT_COLOR,
                    dot_radius=2,
                    skeleton=False,
                )
        return image

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bytes:
        image = self.render(packet, observation, debug_snapshots)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise OverlayEncodingError("failed to encode overlay JPEG")
        return bytes(encoded)

    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
        *,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> bytes | None:
        try:
            image = self.render(packet, observation, debug_snapshots)
            encoded = _encode_bounded_image(image, max_bytes=max_bytes)
            if encoded is not None:
                return encoded
            downscaled = np.asarray(
                cv2.resize(
                    image,
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
            return _encode_bounded_image(downscaled, max_bytes=max_bytes)
        except (cv2.error, OverlayEncodingError):
            return None


def _encode_bounded_image(
    image: NDArray[np.uint8],
    *,
    max_bytes: int,
) -> bytes | None:
    for quality in (85, 70, 55, 40, 25):
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if ok:
            jpeg = bytes(encoded)
            if len(jpeg) <= max_bytes:
                return jpeg
    return None


def _draw_bed_exit_debug(
    image: NDArray[np.uint8],
    snapshot: BedExitDebugSnapshot,
) -> None:
    bed_debug = snapshot.bed_region
    label = "bed_roi"
    if bed_debug is not None:
        label = f"bed_roi:{bed_debug.source}"
        if bed_debug.age_frames is not None:
            label = f"{label} age={bed_debug.age_frames}"
    draw_label(image, label, 8, 18, BED_ROI_TEXT_COLOR, scale=0.5)
    for status in snapshot.statuses:
        color = (
            BED_EXIT_STATUS_COLOR
            if status.occupancy == "exit"
            else BED_PRESENT_STATUS_COLOR
        )
        draw_box(image, status.box, color)
        draw_label(
            image,
            f"bed:{status.occupancy}",
            status.box.x1,
            max(12, status.box.y1 - 4),
            color,
        )


__all__ = [
    "MAX_SNAPSHOT_BYTES",
    "OverlayRenderer",
    "draw_box",
    "draw_caption",
    "draw_label",
    "draw_pose",
    "draw_region",
]
