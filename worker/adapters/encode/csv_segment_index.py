from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import final


@dataclass(frozen=True, slots=True)
class Segment:
    path: Path
    generation: int
    start_time_sec: float
    end_time_sec: float


@final
class CsvSegmentIndex:
    """Track completed segments reported by one FFmpeg CSV segment list."""

    def __init__(self, list_path: Path, generation: int) -> None:
        self._list_path = list_path
        self._generation = generation
        self._segments_by_path: dict[Path, Segment] = {}

    @property
    def list_path(self) -> Path:
        return self._list_path

    @property
    def generation(self) -> int:
        return self._generation

    def refresh(self) -> tuple[Segment, ...]:
        try:
            rows = csv.reader(self._list_path.read_text(encoding="utf-8").splitlines())
        except FileNotFoundError:
            return ()

        discovered: list[Segment] = []
        for row in rows:
            if len(row) != 3:
                continue
            try:
                start_time_sec = float(row[1])
                end_time_sec = float(row[2])
            except ValueError:
                continue
            path = Path(row[0])
            if not path.is_absolute():
                path = self._list_path.parent / path
            if path in self._segments_by_path:
                continue
            segment = Segment(
                path=path,
                generation=self._generation,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
            )
            self._segments_by_path[path] = segment
            discovered.append(segment)
        return tuple(discovered)

    def completed(self) -> tuple[Segment, ...]:
        return tuple(self._segments_by_path.values())

    def select(self, *, start_time_sec: float, end_time_sec: float) -> tuple[Segment, ...]:
        return tuple(
            segment
            for segment in self._segments_by_path.values()
            if segment.end_time_sec >= start_time_sec
            and segment.start_time_sec <= end_time_sec
        )


__all__ = ["CsvSegmentIndex", "Segment"]
