from __future__ import annotations

import math
import random
from typing import Final

import cv2
import numpy as np

from contracts.observation import BoundingBox
from worker.domains.bed_exit.geometry import containment_ratio
from worker.domains.fall.preprocessing import (
    NormalizedPose,
    extract_window_features,
    normalize_pose,
)
from worker.pipeline.perception.features.window_features import (
    extract_window_features as oracle_extract_window_features,
)

_TOLERANCE: Final = 1e-4


def _random_window(
    rng: random.Random, frame_count: int, *, drop_probability: float = 0.0
) -> tuple[NormalizedPose, ...]:
    window: list[NormalizedPose] = []
    for _ in range(frame_count):
        pose: list[tuple[int, int, float]] = []
        for _ in range(17):
            if rng.random() < drop_probability:
                pose.append((rng.randint(0, 640), rng.randint(0, 480), 0.05))
            else:
                pose.append((rng.randint(0, 640), rng.randint(0, 480), rng.uniform(0.3, 1.0)))
        window.append(normalize_pose(tuple(pose), frame_width=640, frame_height=480))
    return tuple(window)


def _falling_window(frame_count: int) -> tuple[NormalizedPose, ...]:
    window: list[NormalizedPose] = []
    for frame_index in range(frame_count):
        base_y = 50 + frame_index * 30
        pose = tuple(
            (100 + keypoint_index, base_y + keypoint_index, 0.9) for keypoint_index in range(17)
        )
        window.append(normalize_pose(pose, frame_width=640, frame_height=480))
    return tuple(window)


def _assert_features_match(window: tuple[NormalizedPose, ...]) -> None:
    domain_features = extract_window_features(window)
    oracle_input = np.asarray(window, dtype=np.float32)
    oracle_features = oracle_extract_window_features(oracle_input)

    assert len(domain_features) == len(oracle_features) == 45
    for index, (domain_value, oracle_value) in enumerate(
        zip(domain_features, oracle_features, strict=True)
    ):
        assert math.isclose(
            domain_value, float(oracle_value), rel_tol=_TOLERANCE, abs_tol=_TOLERANCE
        ), f"feature[{index}] diverged: domain={domain_value!r} oracle={float(oracle_value)!r}"


def test_fall_features_match_oracle_for_steady_pose() -> None:
    rng = random.Random(1)
    _assert_features_match(_random_window(rng, frame_count=8))


def test_fall_features_match_oracle_with_dropped_keypoints() -> None:
    rng = random.Random(2)
    _assert_features_match(_random_window(rng, frame_count=8, drop_probability=0.4))


def test_fall_features_match_oracle_for_simulated_fall_motion() -> None:
    _assert_features_match(_falling_window(frame_count=6))


def test_fall_features_match_oracle_for_single_frame_window() -> None:
    rng = random.Random(3)
    _assert_features_match(_random_window(rng, frame_count=1))


def test_fall_features_match_oracle_across_many_random_windows() -> None:
    rng = random.Random(4)
    for _ in range(25):
        frame_count = rng.randint(1, 12)
        drop_probability = rng.uniform(0.0, 0.6)
        _assert_features_match(
            _random_window(rng, frame_count=frame_count, drop_probability=drop_probability)
        )


_DIAMOND: Final = ((50, 0), (100, 50), (50, 100), (0, 50))
_NOTCHED_L: Final = ((0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100))
_PENTAGON: Final = ((50, 0), (100, 40), (80, 100), (20, 100), (0, 40))
_SELF_INTERSECTING: Final = ((0, 0), (100, 0), (100, 100), (0, 100), (5, -5))

_POLYGONS: Final = (_DIAMOND, _NOTCHED_L, _PENTAGON, _SELF_INTERSECTING)

_EDGE_SAFETY_MARGIN_PX: Final = 4


def _cv2_mask(polygon: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    origin_x, origin_y = min(xs), min(ys)
    width, height = max(xs) - origin_x, max(ys) - origin_y
    mask = np.zeros((height, width), dtype=np.uint8)
    shifted = np.array(
        [[point[0] - origin_x, point[1] - origin_y] for point in polygon],
        dtype=np.int32,
    ).reshape((1, -1, 2))
    cv2.fillPoly(mask, shifted, 1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return mask, origin_x, origin_y


def _cv2_containment_ratio(person: BoundingBox, polygon: tuple[tuple[int, int], ...]) -> float:
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if person_area <= 0:
        return 0.0
    mask, origin_x, origin_y = _cv2_mask(polygon)
    height, width = mask.shape
    left = max(person.x1 - origin_x, 0)
    top = max(person.y1 - origin_y, 0)
    right = min(person.x2 - origin_x, width)
    bottom = min(person.y2 - origin_y, height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = int(mask[top:bottom, left:right].sum())
    return 0.0 if intersection == 0 else intersection / person_area


def _bed(polygon: tuple[tuple[int, int], ...]) -> BoundingBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return BoundingBox(
        x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys), confidence=1.0, polygon=polygon
    )


def _margin_safe_person_box(
    rng: random.Random, polygon: tuple[tuple[int, int], ...]
) -> BoundingBox | None:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x, max_x = min(xs) - 20, max(xs) + 20
    min_y, max_y = min(ys) - 20, max(ys) + 20
    x1 = rng.randint(min_x, max_x - 5)
    y1 = rng.randint(min_y, max_y - 5)
    x2 = x1 + rng.randint(2, 15)
    y2 = y1 + rng.randint(2, 15)
    candidate = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)
    mask, origin_x, origin_y = _cv2_mask(polygon)
    height, width = mask.shape

    def _sum(bx1: int, by1: int, bx2: int, by2: int) -> int:
        left = max(bx1 - origin_x, 0)
        top = max(by1 - origin_y, 0)
        right = min(bx2 - origin_x, width)
        bottom = min(by2 - origin_y, height)
        if right <= left or bottom <= top:
            return 0
        return int(mask[top:bottom, left:right].sum())

    tight_area = max(0, x2 - x1) * max(0, y2 - y1)
    tight_sum = _sum(x1, y1, x2, y2)
    margin = _EDGE_SAFETY_MARGIN_PX
    expanded_sum = _sum(x1 - margin, y1 - margin, x2 + margin, y2 + margin)
    expanded_area = max(0, (x2 - x1) + 2 * margin) * max(0, (y2 - y1) + 2 * margin)
    tight_full = tight_sum == tight_area
    expanded_full = expanded_sum == expanded_area
    tight_empty = tight_sum == 0
    expanded_empty = expanded_sum == 0
    if tight_full and expanded_full:
        return candidate
    if tight_empty and expanded_empty:
        return candidate
    return None


def test_bed_exit_containment_matches_cv2_oracle_across_polygons() -> None:
    rng = random.Random(7)
    for polygon in _POLYGONS:
        bed = _bed(polygon)
        matched = 0
        attempts = 0
        while matched < 20 and attempts < 500:
            attempts += 1
            person = _margin_safe_person_box(rng, polygon)
            if person is None:
                continue
            matched += 1
            domain_ratio = containment_ratio(person, bed)
            oracle_ratio = _cv2_containment_ratio(person, polygon)
            assert math.isclose(domain_ratio, oracle_ratio, rel_tol=_TOLERANCE, abs_tol=1e-6), (
                f"polygon={polygon} person={person!r} domain={domain_ratio} oracle={oracle_ratio}"
            )
        assert matched == 20, f"could not sample enough margin-safe boxes for {polygon!r}"


def test_bed_exit_containment_regression_scenarios() -> None:
    scenarios = (
        (_DIAMOND, BoundingBox(40, 40, 60, 60, 0.9), 1.0),
        (_DIAMOND, BoundingBox(80, 80, 95, 95, 0.9), 0.0),
        (_NOTCHED_L, BoundingBox(10, 10, 50, 50, 0.9), 1.0),
        (_NOTCHED_L, BoundingBox(70, 70, 90, 90, 0.9), 0.0),
        (_SELF_INTERSECTING, BoundingBox(20, 20, 80, 80, 0.9), 1.0),
    )
    for polygon, person, expected in scenarios:
        bed = _bed(polygon)
        assert containment_ratio(person, bed) == expected
        assert _cv2_containment_ratio(person, polygon) == expected
