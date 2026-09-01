#!/usr/bin/env python3
"""Compare ordered, image-free PerceptionFrame diagnostic JSONL timelines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

CONFIDENCE_TOLERANCE = 6e-06
VOLATILE_IDENTITY_KEYS = frozenset({"worker_boot_id", "child_instance_id"})
FORBIDDEN_KEY_PARTS = ("pixel", "tensor", "image")
REQUIRED_ROOT_KEYS = frozenset(
    {"version", "box_source", "identity", "person_box", "human_pose", "bed_region", "association"}
)
REQUIRED_IDENTITY_KEYS = frozenset(
    {"worker_boot_id", "camera_id", "stream_epoch", "seq", "source_pts"}
)
REQUIRED_CHANNELS = {
    "person_box": "boxes",
    "human_pose": "poses",
    "bed_region": "regions",
}
REQUIRED_ASSOCIATION_KEYS = frozenset(
    {"strategy", "track_ids", "selected_cue_indexes", "cue_source", "identity"}
)


class TimelineError(ValueError):
    pass


def _path(parent: str, child: str | int) -> str:
    return f"{parent}[{child}]" if isinstance(child, int) else f"{parent}.{child}"


def _reject_media(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TimelineError(f"{path}: object keys must be strings")
            if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS):
                raise TimelineError(
                    f"{_path(path, key)}: image/pixel/tensor payloads are forbidden"
                )
            _reject_media(child, _path(path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_media(child, _path(path, index))


def _require_keys(value: Any, expected: frozenset[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise TimelineError(f"{path}: expected keys {sorted(expected)}, got {actual}")
    return value


def _validate_identity(value: Any, path: str) -> None:
    identity = _require_keys(value, REQUIRED_IDENTITY_KEYS, path)
    if not isinstance(identity["camera_id"], str):
        raise TimelineError(f"{path}.camera_id: expected string")
    if not isinstance(identity["stream_epoch"], int) or isinstance(identity["stream_epoch"], bool):
        raise TimelineError(f"{path}.stream_epoch: expected integer")
    if not isinstance(identity["seq"], int) or isinstance(identity["seq"], bool):
        raise TimelineError(f"{path}.seq: expected integer")
    if identity["source_pts"] is not None and (
        not isinstance(identity["source_pts"], int) or isinstance(identity["source_pts"], bool)
    ):
        raise TimelineError(f"{path}.source_pts: expected integer or null")


def _validate_frame(value: Any, index: int, box_source: str) -> None:
    path = f"$[{index}]"
    _reject_media(value, path)
    frame = _require_keys(value, REQUIRED_ROOT_KEYS, path)
    if frame["box_source"] != box_source:
        raise TimelineError(f"{path}.box_source: declared source must be {box_source!r}")
    _validate_identity(frame["identity"], f"{path}.identity")
    for channel_name, payload_name in REQUIRED_CHANNELS.items():
        channel = _require_keys(
            frame[channel_name], frozenset({"state", payload_name}), f"{path}.{channel_name}"
        )
        if not isinstance(channel["state"], str) or not isinstance(channel[payload_name], list):
            raise TimelineError(f"{path}.{channel_name}: invalid channel state or payload")
    association = frame["association"]
    if association is not None:
        association = _require_keys(association, REQUIRED_ASSOCIATION_KEYS, f"{path}.association")
        if not isinstance(association["track_ids"], list) or not isinstance(
            association["selected_cue_indexes"], list
        ):
            raise TimelineError(f"{path}.association: cue lists must be arrays")
        _validate_identity(association["identity"], f"{path}.association.identity")


def _read_timeline(path: Path, box_source: str) -> list[Any]:
    frames: list[Any] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise TimelineError(f"{path}:{line_number}: blank lines are not allowed")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TimelineError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                _validate_frame(value, len(frames), box_source)
                frames.append(value)
    except OSError as exc:
        raise TimelineError(f"{path}: {exc}") from exc
    return frames


def _compare(baseline: Any, candidate: Any, path: str, key: str | None = None) -> str | None:
    if key in VOLATILE_IDENTITY_KEYS:
        return None
    if type(baseline) is not type(candidate):
        return path
    if isinstance(baseline, dict):
        if list(baseline) != list(candidate):
            return path
        for child_key in baseline:
            mismatch = _compare(
                baseline[child_key], candidate[child_key], _path(path, child_key), child_key
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(baseline, list):
        if len(baseline) != len(candidate):
            return path
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            mismatch = _compare(left, right, _path(path, index), key)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(baseline, float):
        if key != "confidence" or not math.isfinite(baseline) or not math.isfinite(candidate):
            return path if baseline != candidate else None
        return None if abs(baseline - candidate) <= CONFIDENCE_TOLERANCE else path
    return None if baseline == candidate else path


def _result(baseline_frames: int, candidate_frames: int, mismatch: str | None) -> dict[str, Any]:
    return {
        "baseline_frames": baseline_frames,
        "candidate_frames": candidate_frames,
        "first_mismatch": mismatch,
        "result": "PASS" if mismatch is None else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--box-source", required=True, choices=("pose", "person"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline_count = candidate_count = 0
    mismatch: str | None = None
    try:
        baseline = _read_timeline(args.baseline, args.box_source)
        candidate = _read_timeline(args.candidate, args.box_source)
        baseline_count, candidate_count = len(baseline), len(candidate)
        if baseline_count != candidate_count:
            mismatch = "$: frame count"
        else:
            for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
                mismatch = _compare(left, right, f"$[{index}]")
                if mismatch is not None:
                    break
    except TimelineError as exc:
        mismatch = str(exc)

    report = _result(baseline_count, candidate_count, mismatch)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        parser.error(f"cannot write output: {exc}")
    return 0 if mismatch is None else 1


if __name__ == "__main__":
    sys.exit(main())
