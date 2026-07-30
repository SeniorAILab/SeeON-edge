from __future__ import annotations

from backend.app.features.status.heartbeat_store import NEVER_SEEN, ONLINE, STALE, HeartbeatStore


def test_never_seen_when_in_inventory_without_heartbeat() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    inventory = {"cam-a": {"camera_id": "cam-a", "facility_id": "fac-1"}}

    snap = store.snapshot(inventory, now=1000.0)

    assert snap["cameras"]["cam-a"]["status"] == NEVER_SEEN
    assert snap["cameras"]["cam-a"]["facility_id"] == "fac-1"
    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] is None


def test_online_within_stale_window() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1060.0)

    assert snap["cameras"]["cam-a"]["status"] == ONLINE
    assert snap["cameras"]["cam-a"]["age_sec"] == 60.0


def test_stale_after_window() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1200.0)

    assert snap["cameras"]["cam-a"]["status"] == STALE


def test_seen_camera_outside_inventory_is_reported() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-z", "fac-9", received_at=1000.0)

    snap = store.snapshot({}, now=1010.0)

    assert snap["cameras"]["cam-z"]["status"] == ONLINE
    assert snap["cameras"]["cam-z"]["facility_id"] == "fac-9"


def test_latest_heartbeat_wins() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)
    store.record("cam-a", "fac-1", received_at=1100.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1110.0)

    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] == 1100.0
    assert snap["cameras"]["cam-a"]["age_sec"] == 10.0
