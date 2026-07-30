from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig
from edge.domains.bed_exit.detector import BedExitMonitor, NightWindow
from edge.perception.domain_input import DomainInput
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.config_resolver import resolve_runtime_config
from edge.runtime.edge_worker import _config_refresh
from edge.runtime.edge_worker_config import CameraRuntimeConfig, EdgeWorkerConfig
from edge.runtime.edge_worker_supervisor import EdgeWorkerSupervisor
from edge.runtime.status_store import StatusStore


def box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def observation_for(person: BoundingBox, bed: BoundingBox) -> FrameObservation:
    return FrameObservation(detections=((person,), ()), regions=((bed,), ()))


def domain_input(observation: FrameObservation, time_sec: float) -> DomainInput:
    return DomainInput(
        observation=observation,
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=time_sec,
        frame_index=int(time_sec),
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH, 0),
    )


def runtime_exit_events(
    monitor: BedExitMonitor,
    *,
    bed: BoundingBox,
) -> tuple[dict[str, object], ...]:
    monitor.update(domain_input(observation_for(box(20, 0, 120, 100), bed), 0.0))
    return monitor.update(domain_input(observation_for(box(60, 0, 160, 100), bed), 1.0))


def test_resolve_night_window_prefers_pulled_over_yaml() -> None:
    yaml_config = _yaml_config(night_window={"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})
    pulled = _pulled(PulledNightWindow(start="22:00", end="06:00", tz="UTC"))

    assert resolve_runtime_config(yaml_config, pulled)["bed_exit"] == NightWindow(
        start="22:00",
        end="06:00",
        tz="UTC",
    )


def test_resolve_night_window_falls_back_to_yaml() -> None:
    yaml_config = _yaml_config(night_window={"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})

    assert resolve_runtime_config(yaml_config, None)["bed_exit"] == NightWindow(
        start="21:00",
        end="05:00",
        tz="Asia/Seoul",
    )


def test_resolve_night_window_returns_none_without_pulled_or_yaml_window() -> None:
    assert resolve_runtime_config(_yaml_config(night_window=None), None)["bed_exit"] is None


def test_bed_exit_monitor_live_night_window_flip_without_reconstruction() -> None:
    bed = box(0, 0, 100, 100)
    def clock() -> datetime:
        return datetime(2026, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    monitor = BedExitMonitor(
        min_containment=0.5,
        hold_frames=1,
        grace_frames=0,
        night_window=NightWindow(start="21:00", end="05:00", tz="UTC"),
        clock=clock,
    )

    assert runtime_exit_events(monitor, bed=bed) == ()

    monitor.update_night_window(NightWindow(start="11:00", end="13:00", tz="UTC"))

    assert runtime_exit_events(monitor, bed=bed) == (
        {
            "domain": "bed_exit",
            "event_type": "bed-exit",
            "identity": "0:0",
            "person_id": 0,
            "bed_id": 0,
            "probability": 1.0,
            "time_sec": 1.0,
        },
    )

    monitor.update_night_window(NightWindow(start="21:00", end="05:00", tz="UTC"))
    assert runtime_exit_events(monitor, bed=bed) == ()

    monitor.update_night_window(None)
    assert len(runtime_exit_events(monitor, bed=bed)) == 1


def test_config_refresh_applies_changed_pulled_window_and_keeps_last_on_none() -> None:
    yaml_config = _yaml_config(night_window={"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})
    initial = NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")
    changed = PulledNightWindow(start="11:00", end="13:00", tz="UTC")
    monitor = BedExitMonitor(night_window=initial)
    pulls = iter((_pulled(changed), None))

    refresh = _config_refresh(
        "http://ml-api:8000",
        "token",
        (monitor,),
        yaml_config,
        initial,
        pull_config=lambda _url, _token: next(pulls),
    )

    refresh()
    assert monitor._night_window == NightWindow(start="11:00", end="13:00", tz="UTC")

    refresh()
    assert monitor._night_window == NightWindow(start="11:00", end="13:00", tz="UTC")


def test_config_refresh_keeps_last_on_pull_exception() -> None:
    initial = NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")
    monitor = BedExitMonitor(night_window=initial)

    def fail(_url: str, _token: str | None) -> PulledWorkerConfig | None:
        raise OSError("offline")

    refresh = _config_refresh(
        "http://ml-api:8000",
        "token",
        (monitor,),
        _yaml_config(night_window=None),
        initial,
        pull_config=fail,
    )

    refresh()

    assert monitor._night_window == initial


def test_supervisor_config_refresh_default_none_preserves_current_behavior() -> None:
    supervisor = EdgeWorkerSupervisor.from_workers(
        (_worker("camera-1", _FiniteSource(count=1)),),
        status_store=StatusStore(),
        heartbeat_interval_sec=0.0,
    )

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}


def test_supervisor_invokes_config_refresh_on_heartbeat_cadence() -> None:
    calls = 0

    def config_refresh() -> None:
        nonlocal calls
        calls += 1

    supervisor = EdgeWorkerSupervisor.from_workers(
        (_worker("camera-1", _FiniteSource(count=1)),),
        status_store=StatusStore(),
        heartbeat_interval_sec=0.0,
        config_refresh=config_refresh,
    )

    assert supervisor.run(max_frames_per_camera=1) == {"camera-1": 1}
    assert calls >= 1


class _FiniteSource:
    def __init__(self, count: int = 0) -> None:
        self.count = count

    def __iter__(self) -> Iterator[Frame]:
        for index in range(self.count):
            yield Frame(
                index=index,
                time_sec=float(index),
                image=np.zeros((1, 1, 3), dtype=np.uint8),
            )


def _worker(camera_id: str, source: _FiniteSource) -> CameraWorker:
    return CameraWorker(
        camera_id=camera_id,
        facility_id="facility-1",
        frame_source=source,
        runners={},
    )


def _yaml_config(night_window: dict[str, str] | None) -> EdgeWorkerConfig:
    bed_exit = {"enabled": True}
    if night_window is not None:
        bed_exit["night_window"] = night_window
    return EdgeWorkerConfig(
        relay={"url": "http://127.0.0.1:8000", "token": "relay-token"},
        domains={"bed_exit": bed_exit},
        cameras=(
            CameraRuntimeConfig(
                camera_id="camera-1",
                facility_id="facility-1",
                rtsp_url="rtsp://camera-1.local/trackID=2",
            ),
        ),
    )


def _pulled(night_window: PulledNightWindow | None) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=night_window,
        cameras=(),
    )
