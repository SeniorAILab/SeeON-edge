from __future__ import annotations

from edge.runtime.status_store import CameraStatus, StatusStore


def test_status_store_records_transitions_and_snapshot() -> None:
    store = StatusStore()

    store.set_status("cam-a", "facility-1", CameraStatus.STARTING, timestamp=1.0)
    ready = store.set_status("cam-a", "facility-1", CameraStatus.READY, timestamp=2.0)

    assert ready.status == CameraStatus.READY
    assert store.get_status("cam-a") == ready
    assert store.snapshot()["cameras"]["cam-a"] == {
        "camera_id": "cam-a",
        "facility_id": "facility-1",
        "status": "READY",
        "updated_at": 2.0,
        "error_category": None,
    }


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
    assert store.snapshot()["ops_events"] == [
        {
            "event_type": "model.load_failed",
            "camera_id": "cam-a",
            "facility_id": "facility-1",
            "category": "ModelLoadError",
            "timestamp": 3.0,
            "detail": "weights missing",
        }
    ]
