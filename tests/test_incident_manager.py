from __future__ import annotations

from edge.runtime.incident_manager import IncidentManager


def test_incident_manager_dedupes_duplicate_idempotency_key() -> None:
    manager = IncidentManager(cooldown_sec=10)
    event = {"idempotency_key": "cam-a:fall:1", "time_sec": 1.0}

    assert manager.admit(event)
    assert not manager.admit(event)


def test_incident_manager_suppresses_same_key_within_cooldown_and_readmits_after() -> None:
    manager = IncidentManager(cooldown_sec=5)
    event = {
        "camera_id": "cam-a",
        "domain": "fall",
        "event_type": "fall.detected",
        "identity": "person-1",
    }

    assert manager.admit(event, now_sec=10.0)
    assert not manager.admit(event, now_sec=12.0)
    assert manager.admit(event, now_sec=15.0)
def test_incident_manager_admits_identityless_events_within_cooldown() -> None:
    manager = IncidentManager(cooldown_sec=30)
    event = {"camera_id": "cam-a", "domain": "risk", "event_type": "risk.detected"}

    assert manager.admit(event, now_sec=10.0)
    assert manager.admit(event, now_sec=11.0)


def test_incident_manager_suppresses_fall_counter_churn_within_cooldown() -> None:
    manager = IncidentManager(cooldown_sec=30)
    first = {
        "camera_id": "cam-a",
        "domain": "fall",
        "event_type": "fall",
        "identity": 1,
    }
    next_rising_edge = {**first, "identity": 2}

    assert manager.admit(first, now_sec=100.0)
    assert not manager.admit(next_rising_edge, now_sec=101.0)


def test_incident_manager_readmits_fall_at_exact_cooldown_boundary() -> None:
    manager = IncidentManager(cooldown_sec=30)
    first = {
        "camera_id": "cam-a",
        "domain": "fall",
        "event_type": "fall",
        "identity": 1,
    }
    next_rising_edge = {**first, "identity": 2}

    assert manager.admit(first, now_sec=100.0)
    assert manager.admit(next_rising_edge, now_sec=130.0)


def test_incident_manager_suppresses_bed_exit_track_churn_within_cooldown() -> None:
    manager = IncidentManager(cooldown_sec=30)
    first = {
        "camera_id": "cam-a",
        "domain": "bed_exit",
        "event_type": "bed-exit",
        "bed_id": 1,
        "identity": "person-1:bed-1",
    }
    reassigned_track = {**first, "identity": "person-2:bed-1"}

    assert manager.admit(first, now_sec=100.0)
    assert not manager.admit(reassigned_track, now_sec=101.0)


def test_incident_manager_keeps_distinct_beds_separate() -> None:
    manager = IncidentManager(cooldown_sec=30)
    first = {
        "camera_id": "cam-a",
        "domain": "bed_exit",
        "event_type": "bed-exit",
        "bed_id": 1,
        "identity": "person-1:bed-1",
    }
    other_bed = {**first, "bed_id": 2, "identity": "person-1:bed-2"}

    assert manager.admit(first, now_sec=100.0)
    assert manager.admit(other_bed, now_sec=101.0)


def test_incident_manager_readmits_bed_exit_at_exact_cooldown_boundary() -> None:
    manager = IncidentManager(cooldown_sec=30)
    first = {
        "camera_id": "cam-a",
        "domain": "bed_exit",
        "event_type": "bed-exit",
        "bed_id": 1,
        "identity": "person-1:bed-1",
    }
    reassigned_track = {**first, "identity": "person-2:bed-1"}

    assert manager.admit(first, now_sec=100.0)
    assert manager.admit(reassigned_track, now_sec=130.0)



def test_incident_manager_admits_distinct_keys() -> None:
    manager = IncidentManager(cooldown_sec=30)

    assert manager.admit(
        {
            "camera_id": "cam-a",
            "domain": "fall",
            "event_type": "fall.detected",
            "identity": "person-1",
        }
    )
    assert manager.admit(
        {
            "camera_id": "cam-b",
            "domain": "fall",
            "event_type": "fall.detected",
            "identity": "person-1",
        }
    )
    assert manager.admit(
        {
            "camera_id": "cam-a",
            "domain": "bed_exit",
            "event_type": "bed.exit",
            "identity": "person-1",
        }
    )
