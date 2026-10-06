from __future__ import annotations

from typing import Any

import pytest

from shared.events.replay_wire import (
    MAX_REPLAY_BODY_BYTES,
    MAX_TRACE_FRAME_BYTES,
    MAX_TRACE_FRAMES,
    ReplayTrace,
)
from worker.pipeline.output._mjpeg_http import MAX_REPLAY_BODY_BYTES as WORKER_CAP

_TRUNCATION: dict[str, Any] = {
    "handoff_dropped_frames": 0,
    "pruned_frames": 0,
    "persistence_failed_frames": 0,
    "retention_blocked_frames": 0,
    "oldest_retained_seq": 0,
    "newest_retained_seq": MAX_TRACE_FRAMES - 1,
    "oldest_retained_key": None,
    "newest_retained_key": None,
    "detail_unavailable_reason": None,
}


def _frame(index: int) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{index:06d}",
        "frame_key": f"key-{index:06d}",
        "pts": index * 33_333,
        "source_time": {"value": 1_787_000_000.0 + index * 0.033, "missing_reason": None},
        "frame_width": 1920,
        "frame_height": 1080,
        "bed_region_provenance": "persisted",
        "persons": [
            {
                "ordinal": person,
                "track_id": {"value": person, "missing_reason": None},
                "box": [10.0, 20.0, 30.0, 40.0],
                "confidence": 0.91,
                "keypoints": [
                    {
                        "ordinal": point,
                        "name": f"kp{point}",
                        "x": 1.0 * point,
                        "y": 2.0 * point,
                        "confidence": 0.8,
                    }
                    for point in range(17)
                ],
            }
            for person in range(2)
        ],
        "beds": [
            {
                "ordinal": 0,
                "box": [0.0, 0.0, 100.0, 100.0],
                "confidence": 0.99,
                "provenance": "persisted",
                "polygon": [[float(point), float(point)] for point in range(4)],
            }
        ],
        "components": [
            {
                "ordinal": component,
                "component_id": f"comp-{component}",
                "observation_state": "observed",
            }
            for component in range(3)
        ],
    }


def _timeline(frames: int) -> ReplayTrace:
    return ReplayTrace(
        camera_id="camera-1",
        frames=tuple(_frame(index) for index in range(frames)),
        truncation=dict(_TRUNCATION),
    )


def test_a_fully_retained_timeline_fits_the_transfer_bound() -> None:
    encoded = _timeline(MAX_TRACE_FRAMES).canonical_json().encode()

    assert len(encoded) <= MAX_REPLAY_BODY_BYTES, (
        f"a full {MAX_TRACE_FRAMES}-frame timeline serializes to {len(encoded):,} bytes "
        f"but the transfer bound is {MAX_REPLAY_BODY_BYTES:,}; the long windows replay "
        f"exists for would be refused while short traces succeed"
    )


def test_the_bound_is_derived_from_retention_not_chosen() -> None:
    assert MAX_REPLAY_BODY_BYTES == MAX_TRACE_FRAMES * MAX_TRACE_FRAME_BYTES


def test_both_ends_read_the_same_bound() -> None:
    assert WORKER_CAP == MAX_REPLAY_BODY_BYTES


@pytest.mark.parametrize("frames", [1, 100, 1_000, MAX_TRACE_FRAMES])
def test_timelines_across_the_retention_range_all_fit(frames: int) -> None:
    encoded = _timeline(frames).canonical_json().encode()

    assert len(encoded) <= MAX_REPLAY_BODY_BYTES


def test_the_per_frame_bound_is_not_optimistic() -> None:
    encoded = _timeline(MAX_TRACE_FRAMES).canonical_json().encode()
    observed_per_frame = len(encoded) / MAX_TRACE_FRAMES

    assert observed_per_frame <= MAX_TRACE_FRAME_BYTES, (
        f"a dense frame serializes to {observed_per_frame:,.0f} bytes, above the "
        f"declared {MAX_TRACE_FRAME_BYTES:,}; the derived cap would then be too "
        f"small for a real timeline"
    )


def test_persisted_analysis_recovery_is_retired() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")


def test_product_schema_does_not_grow_runtime_analysis_tables() -> None:
    from backend.app.edge_db.migration.mapping import (
        DIAGNOSTICS_TARGET_TABLES,
        EXPECTED_TARGET_TABLES,
    )

    tables = EXPECTED_TARGET_TABLES | DIAGNOSTICS_TARGET_TABLES
    assert not any(name.startswith("runtime_analysis_") for name in tables)
