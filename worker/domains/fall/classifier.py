from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from contracts.observation import (
    FALL_LABEL_TEXT,
    NORMAL_LABEL_TEXT,
    DetectionLabel,
    FrameObservation,
)
from shared.detection_policies import FALL_POLICY_V1_DEFAULT
from worker.domains.fall.preprocessing import (
    NormalizedPose,
    extract_window_features,
    normalize_pose,
)
from worker.types import DecisionInput, FallModelInput

_KEYPOINT_COUNT = 17
_ZERO_POSE: NormalizedPose = tuple((0.0, 0.0, 0.0) for _ in range(_KEYPOINT_COUNT))


class FallModelMetadataProtocol(Protocol):
    @property
    def window(self) -> int: ...

    @property
    def stride(self) -> int: ...

    @property
    def mode(self) -> Literal["features", "sequence"]: ...


@runtime_checkable
class FallModelProtocol(Protocol):
    @property
    def metadata(self) -> FallModelMetadataProtocol: ...

    @property
    def operating_threshold(self) -> float: ...

    def predict(self, features: FallModelInput) -> float: ...


@dataclass(frozen=True, slots=True)
class FallScoreSnapshot:
    """Exact sequence input and score provenance from the latest classification."""

    track_id: int
    tensor: tuple[tuple[float, ...], ...]
    probability: float
    provenance: Literal["fresh", "cached"]


@dataclass(slots=True)
class FallWindowClassifier:
    model: FallModelProtocol
    operating_threshold: float = FALL_POLICY_V1_DEFAULT.operating_threshold
    _buffers: dict[int, deque[NormalizedPose]] = field(
        default_factory=dict,
        init=False,
    )
    _last_probabilities: dict[int, float] = field(default_factory=dict, init=False)
    _last_tensors: dict[int, tuple[tuple[float, ...], ...]] = field(
        default_factory=dict,
        init=False,
    )
    _frame_counter: int = field(default=0, init=False)
    last_score_snapshots: tuple[FallScoreSnapshot, ...] = field(default=(), init=False)

    def classify(self, input_value: DecisionInput) -> FrameObservation:
        observation = input_value.observation
        live_ids = frozenset(input_value.live_track_ids)
        active_ids: set[int] = set()

        for index, track_id in enumerate(observation.track_ids):
            if track_id is None:
                continue
            active_ids.add(track_id)
            if index < len(observation.keypoints):
                self._buffer_for(track_id).append(
                    normalize_pose(
                        observation.keypoints[index],
                        input_value.frame_width,
                        input_value.frame_height,
                    )
                )

        for track_id in live_ids - active_ids:
            self._buffer_for(track_id).append(_ZERO_POSE)

        for track_id in tuple(self._buffers):
            if track_id not in live_ids:
                del self._buffers[track_id]
                _ = self._last_probabilities.pop(track_id, None)
                _ = self._last_tensors.pop(track_id, None)

        self._frame_counter += 1
        fresh_track_ids: frozenset[int] = frozenset()
        if self._frame_counter % self.model.metadata.stride == 0:
            fresh_track_ids = self._update_due_probabilities()

        labels = tuple(
            self._label_for_track(track_id)
            for track_id in observation.track_ids
            if track_id is not None and track_id in live_ids
        )
        self.last_score_snapshots = tuple(
            FallScoreSnapshot(
                track_id,
                self._last_tensors[track_id],
                self._last_probabilities[track_id],
                "fresh" if track_id in fresh_track_ids else "cached",
            )
            for track_id in observation.track_ids
            if track_id is not None
            and track_id in live_ids
            and track_id in self._last_tensors
            and track_id in self._last_probabilities
        )
        return FrameObservation(
            detections=(observation.boxes, labels),
            poses=observation.poses,
            regions=observation.regions,
            track_ids=observation.track_ids,
        )

    def _buffer_for(self, track_id: int) -> deque[NormalizedPose]:
        existing_buffer = self._buffers.get(track_id)
        if existing_buffer is not None:
            return existing_buffer
        new_buffer: deque[NormalizedPose] = deque(maxlen=self.model.metadata.window)
        self._buffers[track_id] = new_buffer
        return new_buffer

    def _update_due_probabilities(self) -> frozenset[int]:
        metadata = self.model.metadata
        fresh_track_ids: set[int] = set()
        for track_id, buffer in self._buffers.items():
            if len(buffer) < metadata.window:
                continue
            window = tuple(buffer)
            model_input: FallModelInput
            match metadata.mode:
                case "features":
                    model_input = extract_window_features(window)
                case "sequence":
                    model_input = tuple(
                        tuple(coordinate for keypoint in pose for coordinate in keypoint)
                        for pose in window
                    )
                    self._last_tensors[track_id] = model_input
            self._last_probabilities[track_id] = self.model.predict(model_input)
            fresh_track_ids.add(track_id)
        return frozenset(fresh_track_ids)

    def _label_for_track(self, track_id: int) -> DetectionLabel:
        probability = self._last_probabilities.get(track_id, 0.0)
        is_fall = probability >= self.operating_threshold
        return DetectionLabel(
            text=FALL_LABEL_TEXT if is_fall else NORMAL_LABEL_TEXT,
            confidence=probability,
            is_fall=is_fall,
        )


__all__ = [
    "FallModelMetadataProtocol",
    "FallModelProtocol",
    "FallScoreSnapshot",
    "FallWindowClassifier",
]
