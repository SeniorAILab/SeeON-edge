from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Final, Literal

from worker.domains.fall.classifier import FallScoreSnapshot
from worker.runtime.deepstream.fall_diagnostic_writer import FallDiagnosticWriter

_STATES = frozenset(("clear", "fall"))
_SLOT_RE: Final = re.compile(r"^slot-[0-9]{2}$")
_FRAME_COUNT: Final = 35
_PRE_ONSET_FRAMES: Final = 30
_MAX_PERSONS: Final = 16


@dataclass(frozen=True, slots=True)
class FallDiagnosticFrame:
    source_pts: int
    source_seq: int
    native_publish_seq: int
    source_generation: int
    stream_epoch: int
    poses: tuple[tuple[tuple[int, int, float], ...], ...]
    boxes: tuple[tuple[int, int, int, int, float], ...]
    track_ids: tuple[int, ...]
    live_track_ids: tuple[int, ...]
    score: FallScoreSnapshot | None
    previous_state: Literal["clear", "fall"]
    current_state: Literal["clear", "fall"]
    triggered: bool


@dataclass(frozen=True, slots=True)
class FallDiagnosticRecorderStats:
    rejected_frames: int
    skipped_onsets: int
    dropped_bundles: int
    completed_bundles: int
    continuity_resets: int


class FallDiagnosticRecorder:
    """Capture one fixed 30-before/onset/4-after image-free diagnostic bundle."""

    def __init__(
        self,
        camera_slot: str,
        writer: FallDiagnosticWriter,
        *,
        threshold: float = 0.5,
        max_bundles: int = 1,
    ) -> None:
        if _SLOT_RE.fullmatch(camera_slot) is None:
            raise ValueError("camera slot must be anonymous slot-NN")
        if not 0.0 <= threshold <= 1.0 or max_bundles <= 0:
            raise ValueError("invalid diagnostic capture bound")
        self._camera_slot = camera_slot
        self._writer = writer
        self._threshold = threshold
        self._max_bundles = max_bundles
        self._ring: deque[FallDiagnosticFrame] = deque(maxlen=_PRE_ONSET_FRAMES)
        self._active: list[FallDiagnosticFrame] | None = None
        self._rejected_frames = 0
        self._skipped_onsets = 0
        self._dropped_bundles = 0
        self._completed_bundles = 0
        self._continuity_resets = 0
        self._last_identity: tuple[int, int, int, int, int] | None = None

    def record(self, frame: FallDiagnosticFrame) -> None:
        if self._completed_bundles >= self._max_bundles:
            return
        if not _valid_frame(frame):
            self._rejected_frames += 1
            return
        identity = (
            frame.source_generation,
            frame.stream_epoch,
            frame.source_seq,
            frame.source_pts,
            frame.native_publish_seq,
        )
        previous_identity = self._last_identity
        if previous_identity is not None and (
            identity[:2] != previous_identity[:2]
            or identity[2] <= previous_identity[2]
            or identity[3] <= previous_identity[3]
            or identity[4] <= previous_identity[4]
        ):
            self._ring.clear()
            self._active = None
            self._continuity_resets += 1
        self._last_identity = identity
        if self._active is None:
            if not frame.triggered:
                self._ring.append(frame)
                return
            if len(self._ring) < _PRE_ONSET_FRAMES:
                self._skipped_onsets += 1
                self._ring.append(frame)
                return
            self._active = [*self._ring, frame]
        else:
            self._active.append(frame)
        if len(self._active) < _FRAME_COUNT:
            return
        frames = tuple(self._active)
        self._active = None
        self._ring.clear()
        if self._writer.submit_bundle(_bundle_payload(self._camera_slot, self._threshold, frames)):
            self._completed_bundles += 1
        else:
            self._dropped_bundles += 1

    def stats(self) -> FallDiagnosticRecorderStats:
        return FallDiagnosticRecorderStats(
            self._rejected_frames,
            self._skipped_onsets,
            self._dropped_bundles,
            self._completed_bundles,
            self._continuity_resets,
        )


def _valid_frame(frame: FallDiagnosticFrame) -> bool:
    person_count = len(frame.boxes)
    if (
        person_count > _MAX_PERSONS
        or len(frame.poses) != person_count
        or len(frame.track_ids) != person_count
        or frame.previous_state not in _STATES
        or frame.current_state not in _STATES
        or any(len(pose) != 17 for pose in frame.poses)
        or not _finite_nested(frame.poses)
        or not _finite_nested(frame.boxes)
    ):
        return False
    score = frame.score
    return score is None or (
        score.provenance in {"fresh", "cached"}
        and math.isfinite(score.probability)
        and len(score.tensor) == 30
        and all(len(row) == 51 for row in score.tensor)
        and _finite_nested(score.tensor)
    )


def _finite_nested(values: object) -> bool:
    if isinstance(values, (tuple, list)):
        return all(_finite_nested(value) for value in values)
    return (
        isinstance(values, (int, float))
        and not isinstance(values, bool)
        and math.isfinite(values)
    )


def _bundle_payload(
    camera_slot: str,
    threshold: float,
    frames: tuple[FallDiagnosticFrame, ...],
) -> dict[str, object]:
    return {
        "camera_slot": camera_slot,
        "frames": [_frame_payload(frame) for frame in frames],
        "schema_version": 1,
        "threshold": threshold,
    }


def _frame_payload(frame: FallDiagnosticFrame) -> dict[str, object]:
    score = frame.score
    return {
        "boxes": frame.boxes,
        "current_state": frame.current_state,
        "live_track_ids": frame.live_track_ids,
        "native_publish_seq": frame.native_publish_seq,
        "poses": frame.poses,
        "previous_state": frame.previous_state,
        "score": None
        if score is None
        else {
            "probability": score.probability,
            "provenance": score.provenance,
            "tensor_30x51": score.tensor,
            "track_id": score.track_id,
        },
        "source_generation": frame.source_generation,
        "source_pts": frame.source_pts,
        "source_seq": frame.source_seq,
        "stream_epoch": frame.stream_epoch,
        "track_ids": frame.track_ids,
        "triggered": frame.triggered,
    }


__all__ = [
    "FallDiagnosticFrame",
    "FallDiagnosticRecorder",
    "FallDiagnosticRecorderStats",
    "FallScoreSnapshot",
]
