from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from worker.domains.fall.classifier import FallModelInput

_STATES = frozenset(("clear", "fall"))
_SLOT_RE: Final = re.compile(r"^slot-[0-9]{2}$")
_FRAME_COUNT: Final = 35
_FRAME_FIELDS = frozenset(
    (
        "boxes",
        "current_state",
        "live_track_ids",
        "native_publish_seq",
        "poses",
        "previous_state",
        "score",
        "source_generation",
        "source_pts",
        "source_seq",
        "stream_epoch",
        "track_ids",
        "triggered",
    )
)
_SCORE_FIELDS = frozenset(("probability", "provenance", "tensor_30x51", "track_id"))


@dataclass(frozen=True, slots=True)
class FallDiagnosticReplayResult:
    sha256: str
    frame_count: int
    onset_sequences: tuple[int, ...]


def replay_fall_diagnostic_bundle(
    raw: bytes,
    *,
    predict: Callable[[FallModelInput], float],
) -> FallDiagnosticReplayResult:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "camera_slot",
        "frames",
        "schema_version",
        "threshold",
    }:
        raise ValueError("diagnostic bundle fields are not allowlisted")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if canonical != raw or payload["schema_version"] != 1:
        raise ValueError("diagnostic bundle is not canonical schema v1")
    if (
        not isinstance(payload["camera_slot"], str)
        or _SLOT_RE.fullmatch(payload["camera_slot"]) is None
    ):
        raise ValueError("diagnostic camera slot is invalid")
    frames = payload["frames"]
    if not isinstance(frames, list) or len(frames) != _FRAME_COUNT:
        raise ValueError("diagnostic bundle must contain exactly 35 frames")
    threshold = _finite_float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("diagnostic threshold is invalid")
    _validate_frames(frames)
    previous = frames[0]["previous_state"]
    onsets: list[int] = []
    for frame in frames:
        score = frame["score"]
        probability = 0.0
        if score is not None:
            tensor: FallModelInput = tuple(
                tuple(_finite_float(value) for value in row) for row in score["tensor_30x51"]
            )
            probability = predict(tensor)
            if not math.isfinite(probability) or probability != _finite_float(score["probability"]):
                raise ValueError("replayed score differs from captured score")
        current = "fall" if probability >= threshold else "clear"
        triggered = current == "fall" and previous == "clear"
        if (
            frame["previous_state"] != previous
            or frame["current_state"] != current
            or frame["triggered"] is not triggered
        ):
            raise ValueError("replayed latch transition differs from captured transition")
        if triggered:
            onsets.append(frame["source_seq"])
        previous = current
    return FallDiagnosticReplayResult(hashlib.sha256(raw).hexdigest(), len(frames), tuple(onsets))


def _validate_frames(frames: list[object]) -> None:
    previous_identity: tuple[int, int, int, int, int] | None = None
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != _FRAME_FIELDS:
            raise ValueError("diagnostic frame fields are not allowlisted")
        if frame["previous_state"] not in _STATES or frame["current_state"] not in _STATES:
            raise ValueError("diagnostic latch state is invalid")
        if not isinstance(frame["triggered"], bool) or not isinstance(frame["source_seq"], int):
            raise TypeError("diagnostic frame scalar is invalid")
        identity = (
            _strict_int(frame["source_generation"]),
            _strict_int(frame["stream_epoch"]),
            _strict_int(frame["source_seq"]),
            _strict_int(frame["source_pts"]),
            _strict_int(frame["native_publish_seq"]),
        )
        if previous_identity is not None:
            monotonic = zip(identity[2:], previous_identity[2:], strict=True)
            if identity[:2] != previous_identity[:2] or any(
                current <= previous for current, previous in monotonic
            ):
                raise ValueError("diagnostic frame continuity is invalid")
        previous_identity = identity
        score = frame["score"]
        if score is None:
            continue
        if not isinstance(score, dict) or set(score) != _SCORE_FIELDS:
            raise ValueError("diagnostic score fields are not allowlisted")
        if score["provenance"] not in {"fresh", "cached"}:
            raise ValueError("diagnostic score provenance is invalid")
        tensor = score["tensor_30x51"]
        if (
            not isinstance(tensor, list)
            or len(tensor) != 30
            or any(not isinstance(row, list) or len(row) != 51 for row in tensor)
        ):
            raise ValueError("diagnostic score tensor shape is invalid")


def _finite_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("diagnostic numeric value is invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("diagnostic numeric value is non-finite")
    return converted


def _strict_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("diagnostic integer value is invalid")
    return value


__all__ = ["FallDiagnosticReplayResult", "replay_fall_diagnostic_bundle"]
