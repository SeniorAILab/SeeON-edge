from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan
from contracts.frame import Frame
from contracts.runner import bed_result, pose_result
from edge.domains import DOMAIN_REGISTRY
from edge.domains.bed_exit.detector import BedExitMonitor, NightWindow
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker import _RelayClient
from edge.runtime.scheduler import Scheduler
from shared.events.evidence_export_contract import EventReceipt

CAMERA_ID = "camera-night-bed-exit"
FACILITY_ID = "facility-night"
RESIDENT_ID = "resident-night"
RELAY_TOKEN = "relay-token"
NIGHT_WINDOW = NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")


class SpyBackendIngestClient:
    def __init__(self) -> None:
        self.alerts: list[dict] = []
        self.heartbeats = 0

    def send_alert(self, **kwargs) -> bool:
        self.alerts.append(kwargs)
        return True

    def send_alert_receipt(self, **kwargs) -> EventReceipt:
        on_accepted = kwargs.pop("on_accepted")
        on_accepted(datetime.now(tz=ZoneInfo("UTC")).timestamp())
        self.alerts.append(kwargs)
        return EventReceipt("accepted", kwargs["edge_event_id"], "backend-event-1")

    def send_heartbeat(self) -> bool:
        self.heartbeats += 1
        return True


class InProcessRelayClient(_RelayClient):
    def __init__(self, *, client: TestClient) -> None:
        super().__init__(
            alert_url="http://testserver/api/v1/relay/alerts",
            heartbeat_url="http://testserver/api/v1/relay/heartbeat",
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=RESIDENT_ID,
            relay_token=RELAY_TOKEN,
        )
        self._client = client

    def _post(self, url: str, payload: dict[str, object]) -> bool:
        path = url.removeprefix("http://testserver")
        response = self._client.post(
            path,
            json=payload,
            headers={"X-Edge-Relay-Token": self.relay_token},
        )
        if response.status_code >= 400:
            self.failure_count += 1
            return False
        return True


class ScriptedFrameSource:
    def __iter__(self) -> Iterator[Frame]:
        image = np.zeros((120, 180, 3), dtype=np.uint8)
        yield Frame(index=0, time_sec=0.0, image=image)
        yield Frame(index=1, time_sec=1.0, image=image)


class ScriptedPersonRunner:
    def run(self, _frame: np.ndarray):
        if not hasattr(self, "_calls"):
            self._calls = 0
        self._calls += 1
        if self._calls == 1:
            return pose_result(((),), ((10, 10, 90, 80, 0.98),))
        return pose_result(((),), ((52, 10, 132, 80, 0.98),))


class ScriptedBedRunner:
    def run(self, _frame: np.ndarray):
        return bed_result(((0, 0, 90, 100, 0.99),))


def _api_client(backend: SpyBackendIngestClient) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    app.state.camera_inventory = {
        CAMERA_ID: {
            "camera_id": CAMERA_ID,
            "facility_id": FACILITY_ID,
            "resident_id": RESIDENT_ID,
        }
    }
    app.state.backend_ingest_client = backend
    return TestClient(app)


def _clock_at(iso_datetime: str):
    fixed = datetime.fromisoformat(iso_datetime)
    if fixed.tzinfo is None:
        fixed = fixed.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _run_worker_once(*, now: str) -> SpyBackendIngestClient:
    backend = SpyBackendIngestClient()
    api_client = _api_client(backend)
    relay = InProcessRelayClient(client=api_client)
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NIGHT_WINDOW,
        clock=_clock_at(now),
    )
    monitor.registration = DOMAIN_REGISTRY["bed_exit"]
    worker = CameraWorker(
        camera_id=CAMERA_ID,
        facility_id=FACILITY_ID,
        frame_source=ScriptedFrameSource(),
        runners={"pose": ScriptedPersonRunner(), "bed": ScriptedBedRunner()},
        scheduler=Scheduler({"pose": 1, "bed": 1}),
        domain_detectors=(monitor,),
        event_sink=relay,
    )

    assert worker.run(max_frames=2) == 2
    assert relay.failure_count == 0
    return backend


def test_night_bed_exit_reaches_backend_ingest_through_worker_relay_and_ml_api() -> None:
    backend = _run_worker_once(now="2026-06-25T22:00:00+09:00")

    assert backend.alerts == [
        {
            "event_type": "bed-exit",
            "edge_event_id": backend.alerts[0]["edge_event_id"],
            "detected_at": backend.alerts[0]["detected_at"],
            "probability": 1.0,
            "audit": {"config_version": 0, "clock_source": "edge_wall_clock"},
        }
    ]
    assert backend.alerts[0]["detected_at"]


def test_daytime_bed_exit_is_suppressed_before_worker_relay_and_backend_ingest() -> None:
    backend = _run_worker_once(now="2026-06-25T13:00:00+09:00")

    assert backend.alerts == []


def test_disabled_bed_exit_emits_no_alerts_through_relay() -> None:
    # `enabled=false` for the bed-exit domain means no detector is wired, so the
    # worker -> ml-api relay path produces zero alerts even at night.
    backend = SpyBackendIngestClient()
    api_client = _api_client(backend)
    relay = InProcessRelayClient(client=api_client)
    worker = CameraWorker(
        camera_id=CAMERA_ID,
        facility_id=FACILITY_ID,
        frame_source=ScriptedFrameSource(),
        runners={"pose": ScriptedPersonRunner(), "bed": ScriptedBedRunner()},
        scheduler=Scheduler({"pose": 1, "bed": 1}),
        domain_detectors=(),
        event_sink=relay,
    )

    assert worker.run(max_frames=2) == 2
    assert relay.failure_count == 0
    assert backend.alerts == []
