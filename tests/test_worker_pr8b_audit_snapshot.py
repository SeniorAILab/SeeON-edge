from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from edge_worker_fixtures import edge_config_payload

from contracts.frame import Frame
from contracts.observation import FrameObservation
from contracts.runner import RunnerOutput, person_result
from edge.domains.bed_exit.detector import BedExitMonitor
from edge.evidence.snapshot_store import SnapshotStore
from edge.runtime.camera_worker import CameraWorker
from edge.runtime.edge_worker import _domain_detectors, _RelayClient
from edge.runtime.edge_worker_config import EdgeWorkerConfig
from edge.runtime.overlay_renderer import MAX_SNAPSHOT_BYTES, OverlayRenderer
from edge.runtime.scheduler import Scheduler
from shared.events.schemas import CLOCK_SOURCE_EDGE_WALL


class _Runner:
    def run(self, image: np.ndarray) -> RunnerOutput:
        del image
        return person_result(((1, 1, 20, 20, 0.9),))




@dataclass(slots=True)
class _Sink:
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


class _Model:
    version = "fall-model-v1"
    operating_threshold = 0.73



class _SnapshotRenderer:
    def encode_jpeg_bounded(
        self,
        frame: Frame,
        observation: FrameObservation,
        debug_snapshots: tuple[object, ...] = (),
        *,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> bytes | None:
        del frame, observation, debug_snapshots, max_bytes
        return b"jpeg-bytes"


class _FakeHTTPResponse:
    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return b"{}"


def _frame() -> Frame:
    return Frame(index=1, time_sec=123.0, image=np.zeros((64, 64, 3), dtype=np.uint8))


def _worker(
    event_type: str,
    *,
    snapshot_renderer: object | None = None,
    snapshot_store: SnapshotStore | None = None,
) -> tuple[CameraWorker, _Sink]:
    sink = _Sink()
    payload = edge_config_payload(camera_count=1)
    payload["domains"] = {
        "enabled": ["bed_exit" if event_type == "bed-exit" else "fall"]
    }
    detectors = _domain_detectors(
        EdgeWorkerConfig.model_validate(payload), audit_model=_Model()
    )
    detector = next(
        detector
        for detector in detectors
        if (event_type == "bed-exit") is isinstance(detector, BedExitMonitor)
        or (event_type != "bed-exit" and not isinstance(detector, BedExitMonitor))
    )
    detector.update = lambda _domain_input: (  # type: ignore[method-assign]
        {"event_type": event_type, "probability": 0.87},
    )
    worker = CameraWorker(
        "camera-1",
        "facility-1",
        (),
        {"person": _Runner()},
        scheduler=Scheduler({"person": 1}),
        domain_detectors=(detector,),
        event_sink=sink,
        snapshot_renderer=snapshot_renderer,  # type: ignore[arg-type]
        snapshot_store=snapshot_store,
        detector_version="detector-v1",
    )
    return worker, sink


def test_overlay_renderer_encode_jpeg_bounded_returns_size_limited_bytes() -> None:
    jpeg = OverlayRenderer().encode_jpeg_bounded(
        Frame(index=1, time_sec=1.0, image=np.zeros((480, 640, 3), dtype=np.uint8)),
        FrameObservation(),
    )

    assert jpeg is not None
    assert jpeg.startswith(b"\xff\xd8")
    assert len(jpeg) <= MAX_SNAPSHOT_BYTES


def test_overlay_renderer_encode_jpeg_bounded_returns_none_on_encode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> tuple[bool, np.ndarray]:
        del args, kwargs
        raise cv2.error("encode failed")

    monkeypatch.setattr(cv2, "imencode", fail)

    assert OverlayRenderer().encode_jpeg_bounded(_frame(), FrameObservation()) is None


@pytest.mark.parametrize("event_type", ["fall", "bed-exit"])
def test_camera_worker_attaches_audit_and_snapshot_for_alert_events(event_type: str) -> None:
    worker, sink = _worker(
        event_type,
        snapshot_renderer=_SnapshotRenderer(),
    )

    worker.process_frame(_frame())

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["audit"] == {
        "clock_source": CLOCK_SOURCE_EDGE_WALL,
        "model_version": "fall-model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.73,
    }
    assert event["snapshot_jpeg"] == b"jpeg-bytes"


def test_camera_worker_stores_the_rendered_snapshot_bytes(tmp_path: Path) -> None:
    worker, sink = _worker(
        "fall",
        snapshot_renderer=_SnapshotRenderer(),
        snapshot_store=SnapshotStore(tmp_path),
    )

    worker.process_frame(_frame())

    snapshot = sink.events[0]["snapshot"]
    assert isinstance(snapshot, dict)
    path = tmp_path / str(snapshot["path"])
    assert path.read_bytes() == b"jpeg-bytes"
    assert snapshot["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot["snapshot_id"] == sink.events[0]["edge_event_id"]


def test_camera_worker_attaches_audit_without_snapshot_renderer() -> None:
    worker, sink = _worker("fall")

    worker.process_frame(_frame())

    assert sink.events[0]["audit"]["clock_source"] == CLOCK_SOURCE_EDGE_WALL
    assert "snapshot_jpeg" not in sink.events[0]


def test_camera_worker_leaves_non_alert_event_unaffected() -> None:
    worker, sink = _worker(
        "movement-increase",
        snapshot_renderer=_SnapshotRenderer(),
    )

    worker.process_frame(_frame())

    assert "audit" not in sink.events[0]
    assert "snapshot_jpeg" not in sink.events[0]


def test_relay_client_forwards_audit_snapshot_and_keeps_evidence_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        del timeout
        captured.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _RelayClient(
        alert_url="http://127.0.0.1:8000/api/v1/relay/alerts",
        heartbeat_url="http://127.0.0.1:8000/api/v1/relay/heartbeat",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        relay_token="token",
        config_version=7,
    )

    client.emit(
        {
            "event_type": "fall",
            "detected_at": "2026-06-23T12:00:00.000Z",
            "probability": 0.91,
            "audit": {"clock_source": CLOCK_SOURCE_EDGE_WALL, "model_version": "model-v1"},
            "snapshot_jpeg": b"jpeg-bytes",
        }
    )

    payload = captured[0]
    assert payload["audit"] == {
        "clock_source": CLOCK_SOURCE_EDGE_WALL,
        "model_version": "model-v1",
        "config_version": 7,
    }
    assert payload["snapshot_jpeg_base64"] == base64.b64encode(b"jpeg-bytes").decode("ascii")
    assert "audit" not in payload["evidence"]
    assert "snapshot_jpeg" not in payload["evidence"]


def test_relay_client_envelope_less_event_keeps_prior_payload_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeHTTPResponse:
        del timeout
        captured.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _RelayClient(
        alert_url="http://127.0.0.1:8000/api/v1/relay/alerts",
        heartbeat_url="http://127.0.0.1:8000/api/v1/relay/heartbeat",
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id="resident-1",
        relay_token="token",
    )

    client.emit(
        {
            "event_type": "bed-exit",
            "detected_at": "2026-06-23T12:00:00.000Z",
            "probability": 0.91,
        }
    )

    assert captured == [
        {
            "event_type": "bed-exit",
            "probability": 0.91,
            "detected_at": "2026-06-23T12:00:00.000Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "evidence": {
                "event_type": "bed-exit",
                "detected_at": "2026-06-23T12:00:00.000Z",
                "probability": 0.91,
            },
            "resident_id": "resident-1",
        }
    ]
