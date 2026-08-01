from __future__ import annotations

from worker.runtime.telemetry.status_store import StatusStore

# test_status_store_records_transitions_and_snapshot (edge): set_status/get_status/snapshot
# round-trip is superseded by
# tests/test_worker_telemetry_status_store.py::test_status_updates_are_thread_safe_and_camera_scoped
# and ::test_snapshot_preserves_camera_failure_category_and_recovery.


def test_status_store_records_non_secret_ops_event_fields() -> None:
    store = StatusStore()

    event = store.record_ops_event(
        "model.load_failed",
        "cam-a",
        "facility-1",
        "ModelLoadError",
        timestamp=3.0,
        detail="weights missing",
    )

    assert event.event_type == "model.load_failed"
    assert event.camera_id == "cam-a"
    assert event.facility_id == "facility-1"
    assert event.category == "ModelLoadError"
    assert event.timestamp == 3.0
    assert event.detail == "weights missing"
    assert store.snapshot().ops_events == (event,)
