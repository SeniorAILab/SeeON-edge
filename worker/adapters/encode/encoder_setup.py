from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from worker.adapters.encode.adapter_errors import EncoderPolicyError
from worker.adapters.encode.models import EncodePolicy, EncoderGeometry


@dataclass(frozen=True, slots=True)
class SessionRequest:
    camera: str
    encoder: EncodePolicy
    geometry: EncoderGeometry
    generation: int


def parse_encode_policy(value: str) -> EncodePolicy:
    match value:
        case "h264_nvenc":
            return "h264_nvenc"
        case "libx264":
            return "libx264"
        case _:
            raise EncoderPolicyError(f"unsupported profile encoder: {value}")


def camera_directory_name(camera: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("._") or "camera"
    digest = hashlib.sha256(camera.encode()).hexdigest()[:12]
    return f"{readable[:48]}-{digest}"


__all__ = ["SessionRequest", "camera_directory_name", "parse_encode_policy"]
