"""Mutable greedy-IoU tracker for perception observations."""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.observation import BoundingBox
from edge.features.geometry import greedy_match, iou  # noqa: F401


@dataclass
class _Track:
    track_id: int
    last_box: BoundingBox
    misses: int = field(default=0)


class GreedyIouTracker:
    """Greedy IoU tracker that assigns persistent track ids to bounding boxes.

    On each ``update(boxes)`` call:

    1. Compute IoU between every live track's ``last_box`` and every incoming box.
    2. Sort all (track, box) pairs by IoU descending.
    3. Greedily match pairs whose IoU >= ``min_iou``; each track and box may only
       be used in one match.
    4. Unmatched boxes are assigned fresh, auto-incrementing ids.
    5. Unmatched tracks accumulate ``misses``; those whose ``misses > max_misses``
       are evicted.  A missed track's ``last_box`` is kept for future matching
       while it survives the TTL window.
    """

    def __init__(self, min_iou: float = 0.3, max_misses: int = 30) -> None:
        self._min_iou = min_iou
        self._max_misses = max_misses
        self._tracks: list[_Track] = []
        self._next_id: int = 0

    @property
    def live_ids(self) -> frozenset[int]:
        """Frozenset of track ids that are currently alive (not yet evicted)."""
        return frozenset(t.track_id for t in self._tracks)

    def update(self, boxes: tuple[BoundingBox, ...]) -> tuple[int, ...]:
        """Associate *boxes* with live tracks; return one track id per box (index-aligned)."""
        # Snapshot existing tracks before appending new ones this frame.
        existing = list(self._tracks)
        matches = greedy_match(
            tuple(track.last_box for track in existing),
            boxes,
            self._min_iou,
        )

        matched_track_idxs: set[int] = set()
        matched_box_idxs: set[int] = set()
        box_to_id: dict[int, int] = {}

        for ti, bi in matches:
            matched_track_idxs.add(ti)
            matched_box_idxs.add(bi)
            box_to_id[bi] = existing[ti].track_id
            existing[ti].last_box = boxes[bi]
            existing[ti].misses = 0

        # Fresh ids for unmatched boxes.
        new_tracks: list[_Track] = []
        for bi in range(len(boxes)):
            if bi not in matched_box_idxs:
                new_id = self._next_id
                self._next_id += 1
                box_to_id[bi] = new_id
                new_tracks.append(_Track(track_id=new_id, last_box=boxes[bi]))

        # Accumulate misses for unmatched existing tracks; evict if needed.
        surviving_existing: list[_Track] = []
        for ti, track in enumerate(existing):
            if ti in matched_track_idxs:
                surviving_existing.append(track)
            else:
                track.misses += 1
                if track.misses <= self._max_misses:
                    surviving_existing.append(track)
                # else: evicted — drop silently

        self._tracks = surviving_existing + new_tracks

        return tuple(box_to_id[bi] for bi in range(len(boxes)))
