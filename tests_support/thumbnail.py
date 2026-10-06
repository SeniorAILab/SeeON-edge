from __future__ import annotations

from pathlib import Path


class DeterministicThumbnailGenerator:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error

    def generate(self, video_path: Path, thumbnail_path: Path, duration_s: float) -> Path:
        if self._error is not None:
            raise self._error
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail_path.write_bytes(f"thumbnail:{video_path.name}:{duration_s:g}".encode())
        return thumbnail_path


__all__ = ["DeterministicThumbnailGenerator"]
