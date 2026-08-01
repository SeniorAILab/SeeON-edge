# pyright: reportAny=false
from __future__ import annotations

import math
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD

Pose: TypeAlias = tuple[tuple[int, int, float], ...]

_KEYPOINT_COUNT: Final = 17
_KEYPOINT_DIMENSIONS: Final = 3
_LEFT_SHOULDER: Final = 5
_RIGHT_SHOULDER: Final = 6
_LEFT_HIP: Final = 11
_RIGHT_HIP: Final = 12
_FEATURE_DIMENSION: Final = 45


def normalize_pose(
    pose: Pose, frame_width: int, frame_height: int
) -> NDArray[np.float32]:
    normalized = np.zeros(
        (_KEYPOINT_COUNT, _KEYPOINT_DIMENSIONS),
        dtype=np.float32,
    )
    for index, (x_coordinate, y_coordinate, confidence) in enumerate(pose):
        if index >= _KEYPOINT_COUNT:
            break
        if confidence >= DEFAULT_FALL_CONFIDENCE_THRESHOLD:
            normalized[index] = (
                float(x_coordinate) / frame_width,
                float(y_coordinate) / frame_height,
                float(confidence),
            )
    return normalized


def extract_window_features(
    window: NDArray[np.float32],
) -> NDArray[np.float32]:
    values: NDArray[np.float32] = np.asarray(window, dtype=np.float32)
    frame_count = len(values)
    x_coordinates: NDArray[np.float32] = values[:, :, 0]
    y_coordinates: NDArray[np.float32] = values[:, :, 1]
    confidence: NDArray[np.float32] = values[:, :, 2]
    valid: NDArray[np.bool_] = confidence > DEFAULT_FALL_CONFIDENCE_THRESHOLD
    features: list[float] = []

    valid_step: NDArray[np.bool_] = valid[:-1] & valid[1:]
    absolute_x_delta: NDArray[np.float32] = np.abs(
        np.diff(x_coordinates, axis=0)
    )
    absolute_y_delta: NDArray[np.float32] = np.abs(
        np.diff(y_coordinates, axis=0)
    )
    for keypoint_index in range(_KEYPOINT_COUNT):
        mask: NDArray[np.bool_] = valid_step[:, keypoint_index]
        if mask.any():
            selected_x: NDArray[np.float32] = absolute_x_delta[:, keypoint_index][
                mask
            ]
            selected_y: NDArray[np.float32] = absolute_y_delta[:, keypoint_index][
                mask
            ]
            features.extend(
                (
                    _mean(selected_x),
                    _mean(selected_y),
                )
            )
        else:
            features.extend((0.0, 0.0))

    summed_valid: NDArray[np.int64] = valid.sum(axis=1)
    valid_counts: NDArray[np.float32] = summed_valid.astype(np.float32)
    denominator: NDArray[np.float32] = np.maximum(valid_counts, 1.0)
    weighted_x: NDArray[np.float32] = x_coordinates * valid
    weighted_y: NDArray[np.float32] = y_coordinates * valid
    weighted_x_sums: NDArray[np.float32] = weighted_x.sum(axis=1)
    weighted_y_sums: NDArray[np.float32] = weighted_y.sum(axis=1)
    centroid_x: NDArray[np.float32] = np.where(
        valid_counts > 0,
        weighted_x_sums / denominator,
        0.0,
    )
    centroid_y: NDArray[np.float32] = np.where(
        valid_counts > 0,
        weighted_y_sums / denominator,
        0.0,
    )
    step_distances: NDArray[np.float32] = np.sqrt(
        np.diff(centroid_x) ** 2 + np.diff(centroid_y) ** 2
    )
    features.extend(
        (
            _sum(step_distances),
            _max(step_distances) if len(step_distances) > 0 else 0.0,
        )
    )

    aspect_ratios: list[float] = []
    for frame_index in range(frame_count):
        frame_mask: NDArray[np.bool_] = valid[frame_index]
        frame_valid_count: np.int64 = frame_mask.sum()
        if frame_valid_count >= 2:
            frame_x: NDArray[np.float32] = x_coordinates[frame_index, frame_mask]
            frame_y: NDArray[np.float32] = y_coordinates[frame_index, frame_mask]
            width = _max(frame_x) - _min(frame_x)
            height = _max(frame_y) - _min(frame_y)
            aspect_ratios.append(width / height if height > 0.0 else 0.0)
    if aspect_ratios:
        ratios: NDArray[np.float32] = np.asarray(aspect_ratios, dtype=np.float32)
        features.extend(
            (
                _mean(ratios),
                _std(ratios),
                _max(ratios) - _min(ratios),
            )
        )
    else:
        features.extend((0.0, 0.0, 0.0))

    tilt_values: list[float] = []
    torso_vertical_values: list[float] = []
    for frame_index in range(frame_count):
        shoulder_x = [
            _at(x_coordinates, frame_index, index)
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[frame_index, index]
        ]
        shoulder_y = [
            _at(y_coordinates, frame_index, index)
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[frame_index, index]
        ]
        hip_x = [
            _at(x_coordinates, frame_index, index)
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[frame_index, index]
        ]
        hip_y = [
            _at(y_coordinates, frame_index, index)
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[frame_index, index]
        ]
        if shoulder_x and hip_x:
            x_delta = sum(hip_x) / len(hip_x) - sum(shoulder_x) / len(shoulder_x)
            y_delta = sum(hip_y) / len(hip_y) - sum(shoulder_y) / len(shoulder_y)
            magnitude = math.sqrt(x_delta * x_delta + y_delta * y_delta)
            if magnitude > 0.0:
                tilt_values.append(abs(x_delta) / magnitude)
            torso_vertical_values.append(
                abs(sum(shoulder_y) / len(shoulder_y) - sum(hip_y) / len(hip_y))
            )
    features.append(_mean_values(tilt_values) if tilt_values else 0.0)

    absolute_centroid_y_delta: NDArray[np.float32] = np.abs(
        np.diff(centroid_y)
    )
    features.extend(
        (
            _mean(absolute_centroid_y_delta)
            if len(absolute_centroid_y_delta) > 0
            else 0.0,
            _max(absolute_centroid_y_delta)
            if len(absolute_centroid_y_delta) > 0
            else 0.0,
        )
    )

    if torso_vertical_values:
        torso_vertical: NDArray[np.float32] = np.asarray(
            torso_vertical_values,
            dtype=np.float32,
        )
        features.extend(
            (_mean(torso_vertical), _std(torso_vertical))
        )
    else:
        features.extend((0.0, 0.0))
    squared_step_distances: NDArray[np.float32] = step_distances**2
    features.append(_sum(squared_step_distances))

    result: NDArray[np.float32] = np.asarray(features, dtype=np.float32)
    if result.shape != (_FEATURE_DIMENSION,):
        message = (
            f"fall feature shape must be ({_FEATURE_DIMENSION},), "
            f"received {result.shape}"
        )
        raise RuntimeError(message)
    return result


def _at(values: NDArray[np.float32], row: int, column: int) -> float:
    value: np.float32 = values[row, column]
    return float(value)


def _mean(values: NDArray[np.float32]) -> float:
    value: np.float32 = values.mean()
    return float(value)


def _mean_values(values: list[float]) -> float:
    return float(np.mean(values))


def _sum(values: NDArray[np.float32]) -> float:
    value: np.float32 = values.sum()
    return float(value)


def _max(values: NDArray[np.float32]) -> float:
    value: np.float32 = values.max()
    return float(value)


def _min(values: NDArray[np.float32]) -> float:
    value: np.float32 = values.min()
    return float(value)


def _std(values: NDArray[np.float32]) -> float:
    value: np.float32 = values.std()
    return float(value)


__all__ = ["extract_window_features", "normalize_pose"]
