from __future__ import annotations

from worker.runtime.telemetry.status_store import StatusStore


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
