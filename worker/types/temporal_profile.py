from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Final


class TemporalProfileError(ValueError):
    ...


_CURRENT_INGEST_FPS: Final = 30.0
_CURRENT_POSE_FPS: Final = 15.0
_CURRENT_BED_INTERVAL_FRAMES: Final = 180
_CURRENT_BED_DECISION_HZ: Final = _CURRENT_INGEST_FPS / _CURRENT_BED_INTERVAL_FRAMES


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalProfileError(f"{name} must be a finite number > 0")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise TemporalProfileError(f"{name} must be a finite number > 0")
    return numeric


def _default_decision_hz() -> dict[str, float]:
    return {"bed": _CURRENT_BED_DECISION_HZ}


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    ingest_fps: float
    pose_fps: float | None = None
    decision_hz: Mapping[str, float] = field(default_factory=_default_decision_hz)

    def __post_init__(self) -> None:
        ingest_fps = _require_positive_finite("ingest_fps", self.ingest_fps)
        object.__setattr__(self, "ingest_fps", ingest_fps)
        if self.pose_fps is None:
            object.__setattr__(self, "pose_fps", ingest_fps)
        else:
            object.__setattr__(
                self,
                "pose_fps",
                _require_positive_finite("pose_fps", self.pose_fps),
            )
        normalized: dict[str, float] = {}
        for name, hz in self.decision_hz.items():
            if not name:
                raise TemporalProfileError("decision_hz keys must be non-empty")
            normalized[name] = _require_positive_finite(f"decision_hz[{name!r}]", hz)
        object.__setattr__(self, "decision_hz", MappingProxyType(normalized))

    @property
    def target_fps(self) -> float:
        return self.ingest_fps

    @property
    def frame_interval_sec(self) -> float:
        return 1.0 / self.ingest_fps

    def pose_interval_frames(self, *, frame_stride: int = 1) -> int:
        if type(frame_stride) is not int or frame_stride <= 0:
            raise TemporalProfileError("frame_stride must be an integer > 0")
        pose_fps = self.ingest_fps if self.pose_fps is None else self.pose_fps
        return max(1, round(self.ingest_fps / pose_fps)) * frame_stride

    def decision_interval_frames(self, domain: str) -> int:
        try:
            hz = self.decision_hz[domain]
        except KeyError as exc:
            raise TemporalProfileError(f"no decision_hz declared for {domain!r}") from exc
        return max(1, round(self.ingest_fps / hz))

    def task_intervals(self, *, frame_stride: int = 1) -> dict[str, int]:
        intervals = {"pose": self.pose_interval_frames(frame_stride=frame_stride)}
        for domain in self.decision_hz:
            intervals[domain] = self.decision_interval_frames(domain)
        return intervals


CURRENT_TEMPORAL_PROFILE: Final = TemporalProfile(
    ingest_fps=_CURRENT_INGEST_FPS, pose_fps=_CURRENT_POSE_FPS
)


__all__ = [
    "CURRENT_TEMPORAL_PROFILE",
    "TemporalProfile",
    "TemporalProfileError",
]
