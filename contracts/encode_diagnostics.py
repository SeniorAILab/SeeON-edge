from __future__ import annotations

from dataclasses import dataclass

ENCODE_BACKENDS = ("h264_nvenc", "libx264")

ENCODE_FALLBACK_REASONS = (
    "nvenc_probe_failed",
    "session_open_failed",
)


@dataclass(frozen=True, slots=True)
class EncodeSelection:
    requested: str
    selected: str | None
    fallback_count: int
    last_reason: str | None
    updated_at_sec: float

    def __post_init__(self) -> None:
        if self.requested not in ENCODE_BACKENDS:
            raise ValueError(f"unsupported encode backend: {self.requested}")
        if self.selected is not None and self.selected not in ENCODE_BACKENDS:
            raise ValueError(f"unsupported encode backend: {self.selected}")


__all__ = ["ENCODE_BACKENDS", "ENCODE_FALLBACK_REASONS", "EncodeSelection"]
