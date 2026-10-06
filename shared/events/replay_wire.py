from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final


class ReplayWireError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    camera_id: str
    frames: tuple[dict[str, object], ...]
    truncation: dict[str, object]

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ReplayWireError("camera_id is required")
        if not self.frames:
            raise ReplayWireError("replay requires at least one captured frame")
        required_truncation = {
            "handoff_dropped_frames",
            "pruned_frames",
            "persistence_failed_frames",
            "retention_blocked_frames",
            "oldest_retained_seq",
            "newest_retained_seq",
            "oldest_retained_key",
            "newest_retained_key",
            "detail_unavailable_reason",
        }
        if set(self.truncation) != required_truncation:
            raise ReplayWireError("truncation fields are incomplete")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        decoded: dict[str, object] = json.loads(self.canonical_json())
        return decoded


def decode_replay_trace(payload: object) -> ReplayTrace:
    if not isinstance(payload, dict) or set(payload) != {"camera_id", "frames", "truncation"}:
        raise ReplayWireError("replay payload has undeclared or missing fields")
    camera_id = payload["camera_id"]
    frames = payload["frames"]
    truncation = payload["truncation"]
    if (
        not isinstance(camera_id, str)
        or not isinstance(frames, list)
        or not isinstance(truncation, dict)
    ):
        raise ReplayWireError("replay payload has invalid field types")
    normalized_frames: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ReplayWireError("frame must be an object")
        normalized_frames.append(frame)
    return ReplayTrace(camera_id, tuple(normalized_frames), truncation)


MAX_TRACE_FRAMES: Final = 3_000

MAX_TRACE_FRAME_BYTES: Final = 6 * 1024

MAX_REPLAY_BODY_BYTES: Final = MAX_TRACE_FRAMES * MAX_TRACE_FRAME_BYTES

__all__ = [
    "MAX_REPLAY_BODY_BYTES",
    "MAX_TRACE_FRAMES",
    "MAX_TRACE_FRAME_BYTES",
    "ReplayTrace",
    "ReplayWireError",
    "decode_replay_trace",
]
