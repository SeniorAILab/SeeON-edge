from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

from worker.types.preview import OverlaySelection

RELAY_TOKEN_HEADER: Final = "X-Edge-Relay-Token"

MJPEG_BOUNDARY: Final = b"frame"
MJPEG_MEDIA_TYPE: Final = f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}"

PROBE_PATH: Final = "/probe"
REPLAY_PATH: Final = "/replay"

STREAM_PREFIX: Final = "/stream/"
SNAPSHOT_PREFIX: Final = "/snapshot/"
OVERLAY_PREFIX: Final = "/overlay/"
POSE_SUFFIX: Final = "/pose"
BED_ZONE_SUFFIX: Final = "/bed-zone/recognize"


def stream_camera_id(path: str) -> str | None:
    if not path.startswith(STREAM_PREFIX):
        return None
    return unquote(path[len(STREAM_PREFIX) :])


def snapshot_camera_id(path: str) -> str | None:
    if not path.startswith(SNAPSHOT_PREFIX):
        return None
    return unquote(path[len(SNAPSHOT_PREFIX) :])


def pose_camera_id(path: str) -> str | None:
    return _overlay_camera_id(path, POSE_SUFFIX)


def bed_zone_camera_id(path: str) -> str | None:
    return _overlay_camera_id(path, BED_ZONE_SUFFIX)


def _overlay_camera_id(path: str, suffix: str) -> str | None:
    if not path.startswith(OVERLAY_PREFIX) or not path.endswith(suffix):
        return None
    camera_id = unquote(path[len(OVERLAY_PREFIX) : -len(suffix)])
    return camera_id or None


def parse_overlay_selection(payload: object) -> OverlaySelection | None:
    if not isinstance(payload, Mapping) or set(payload) != {"person", "bed"}:
        return None
    person = payload["person"]
    bed = payload["bed"]
    if not isinstance(person, bool) or not isinstance(bed, bool):
        return None
    return OverlaySelection(person=person, bed=bed)


def overlay_selection_body(selection: OverlaySelection) -> dict[str, bool]:
    return {"person": selection.person, "bed": selection.bed}


ProbeErrorClass: TypeAlias = Literal["auth", "timeout", "decode", "unsupported", "unavailable"]


def normalize_probe_error_class(value: object) -> ProbeErrorClass:
    if value == "auth":
        return "auth"
    if value == "timeout":
        return "timeout"
    if value == "unsupported":
        return "unsupported"
    if value == "unavailable":
        return "unavailable"
    return "decode"


def parse_probe_request(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    rtsp_url = payload.get("rtsp_url")
    if not isinstance(rtsp_url, str) or rtsp_url.strip() == "":
        return None
    return rtsp_url


@dataclass(frozen=True, slots=True)
class ProbeResponse:
    ok: bool
    error_class: ProbeErrorClass | None = None
    backend: str | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def sanitized(cls, payload: Mapping[str, object]) -> ProbeResponse:
        if payload.get("ok") is not True:
            return cls(
                ok=False, error_class=normalize_probe_error_class(payload.get("error_class"))
            )
        backend = payload.get("backend")
        width = payload.get("width")
        height = payload.get("height")
        return cls(
            ok=True,
            backend=backend if isinstance(backend, str) else None,
            width=width if isinstance(width, int) else None,
            height=height if isinstance(height, int) else None,
        )

    def as_dict(self) -> dict[str, bool | str | int]:
        if not self.ok:
            return {"ok": False, "error_class": normalize_probe_error_class(self.error_class)}
        payload: dict[str, bool | str | int] = {"ok": True}
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        return payload


BED_ZONE_NOT_FOUND_BODY: Final = {"error_class": "bed_not_found"}
DEFAULT_BED_ZONE_CONFIDENCE: Final = 0.25
MIN_BED_ZONE_CONFIDENCE: Final = 0.05
MAX_BED_ZONE_CONFIDENCE: Final = 0.95


def parse_bed_zone_recognize_request(payload: object) -> float | None:
    if not isinstance(payload, dict) or not payload.keys() <= {"confidence"}:
        return None
    confidence = payload.get("confidence", DEFAULT_BED_ZONE_CONFIDENCE)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(float(confidence))
    ):
        return None
    parsed = float(confidence)
    if not MIN_BED_ZONE_CONFIDENCE <= parsed <= MAX_BED_ZONE_CONFIDENCE:
        return None
    return parsed


@dataclass(frozen=True, slots=True)
class BedZoneRecognizeRegion:
    id: str
    polygon: tuple[tuple[int, int], ...]
    origin: Literal["model"] = "model"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "polygon": [[x, y] for x, y in self.polygon],
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class BedZoneRecognizeResponse:
    regions: tuple[BedZoneRecognizeRegion, ...]
    image_width: int
    image_height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "regions": [region.as_dict() for region in self.regions],
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


__all__ = [
    "BED_ZONE_NOT_FOUND_BODY",
    "DEFAULT_BED_ZONE_CONFIDENCE",
    "MAX_BED_ZONE_CONFIDENCE",
    "MIN_BED_ZONE_CONFIDENCE",
    "MJPEG_BOUNDARY",
    "MJPEG_MEDIA_TYPE",
    "PROBE_PATH",
    "RELAY_TOKEN_HEADER",
    "REPLAY_PATH",
    "BedZoneRecognizeRegion",
    "BedZoneRecognizeResponse",
    "ProbeErrorClass",
    "ProbeResponse",
    "bed_zone_camera_id",
    "normalize_probe_error_class",
    "overlay_selection_body",
    "parse_bed_zone_recognize_request",
    "parse_overlay_selection",
    "parse_probe_request",
    "pose_camera_id",
    "snapshot_camera_id",
    "stream_camera_id",
]
