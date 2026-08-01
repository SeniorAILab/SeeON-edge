"""Composition of the evidence egress path: outbox -> relay HTTP, and the
real per-camera clip recorder wiring that replaces ``_NullClipRecorder``.

Covers the properties the composition root relies on but does not itself
prove: an event already staged in the durable outbox actually reaches the
relay's ``/api/v1/relay/alerts`` endpoint once ``WorkerRuntime.run()`` starts
the export sender, ``stop()`` really reaps that sender's background thread
instead of leaking it, and per-camera clip recorder composition hands each
camera a distinct object while sharing the one real encoder/actor underneath
(the existing ``ClipRecorder`` design, not a new policy).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import shared.events.evidence_export_client as evidence_export_client_module
import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId, StagedEvent
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FakeRunner:
    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(
        self,
        task: str,
        **_options: str | int | float | bool | None,
    ) -> _FakeRunner:
        runner = _FakeRunner(task)
        self.created.append((task, runner))
        return runner


@final
class _FakeClipRecorder:
    """Stands in for the real ``ClipRecorder``: ``_build_clip_frame_feeder``
    composition wiring only needs an object identity to bind, not real
    encode behavior."""

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        return True


@final
class _NoOpPump:
    """Composition tests assert on wiring, not pump throughput."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.stop_count = 0

    def run(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_count += 1


def _pump_factory(
    camera: CameraRuntimeConfig,
    _bus: object,
    _analytics: object,
    _decision: object,
    _sink: object,
) -> _NoOpPump:
    return _NoOpPump(camera.camera_id)


@final
class _InstantLoop:
    """Ingest loop fake that reports ready and returns immediately."""

    def __init__(self, camera_id: str, reporter: IngestReporter) -> None:
        self.camera_id = camera_id
        self._reporter = reporter
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self._reporter.mark_starting(self.camera_id)
        self._reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        self.stop_count += 1


@final
class _InstantLoopFactory:
    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _InstantLoop:
        return _InstantLoop(camera.camera_id, reporter)


@final
class _DeliveryWaitingLoop:
    """Ingest loop fake that blocks camera-a until export delivery is observed.

    Without this, ``IngestSupervisor.join()`` (and therefore ``stop()``,
    which reaps the export sender thread) could race ahead of the sender's
    first delivery attempt, making the delivery assertion flaky.
    """

    def __init__(
        self, camera_id: str, reporter: IngestReporter, delivered: threading.Event
    ) -> None:
        self.camera_id = camera_id
        self._reporter = reporter
        self._delivered = delivered
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self._reporter.mark_starting(self.camera_id)
        assert self._delivered.wait(timeout=5.0), "evidence export never delivered the event"
        self._reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        self.stop_count += 1


@final
class _DeliveryWaitingLoopFactory:
    def __init__(self, delivered: threading.Event) -> None:
        self._delivered = delivered

    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _DeliveryWaitingLoop:
        return _DeliveryWaitingLoop(camera.camera_id, reporter, self._delivered)


def _stub_heartbeat_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(
        _url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", request)


def _runtime(
    config: WorkerConfig,
    serving: _FakeServingClient,
    loop_factory: Callable[..., object],
    state_dir: Path,
    *,
    hard_exit: Callable[[int], None] = lambda _code: None,
    clip_recorder_factory: Callable[[CameraRuntimeConfig], object] | None = None,
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        loop_factory=loop_factory,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=hard_exit,
        clip_recorder_factory=clip_recorder_factory,
    )


def _config(*camera_ids: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": f"facility-{camera_id.removeprefix('camera-')}",
                    "rtsp_url": f"rtsp://example.test/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
                for camera_id in camera_ids
            ],
        }
    )


_EDGE_EVENT_ID = EdgeEventId("11111111-1111-4111-8111-111111111111")


def _stage_one_admitted_event(outbox_path: Path) -> str:
    payload_json = json.dumps(
        {
            "edge_event_id": _EDGE_EVENT_ID,
            "event_type": "fall_detected",
            "probability": 0.9,
            "detected_at": "2026-08-01T00:00:00Z",
            "camera_id": "camera-a",
            "facility_id": "facility-a",
            "evidence": {},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    with EvidenceOutbox.open(outbox_path) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=_EDGE_EVENT_ID,
                detected_at="2026-08-01T00:00:00Z",
                payload_json=payload_json,
                queued_at=1.0,
            )
        )
        # Mirrors EvidenceEventSink.emit(): stage() alone leaves the row in
        # the non-claimable STAGED state; complete() (here: no clip bound)
        # transitions it to READY, the only state EvidenceSender.claim() polls.
        outbox.mark_ready(_EDGE_EVENT_ID)
    return payload_json


def _enable_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ML_WORKER_EVENT_CLIP_EXPORT_ENABLED", "1")
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    outbox_path = tmp_path / "evidence-outbox.sqlite3"
    monkeypatch.setenv("ML_WORKER_EVIDENCE_OUTBOX_PATH", str(outbox_path))
    return outbox_path


def _fake_capabilities_response() -> HttpResult:
    return 200, {}, json.dumps({"event_idempotency": 1, "clip_export": 0}).encode()


def _fake_event_accepted_response() -> HttpResult:
    return (
        200,
        {},
        json.dumps(
            {"status": "accepted", "edge_event_id": _EDGE_EVENT_ID, "event_id": "backend-1"}
        ).encode(),
    )


def test_admitted_event_staged_in_the_outbox_is_delivered_to_the_relay_alerts_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wired lifecycle: outbox row -> EvidenceSender -> RelayEvidenceClient ->
    a real HTTP-transport-level fake, asserting the actual POST URL/payload."""
    _stub_heartbeat_transport(monkeypatch)
    outbox_path = _enable_export(monkeypatch, tmp_path)
    _stage_one_admitted_event(outbox_path)

    posts: list[tuple[str, bytes | None]] = []
    delivered = threading.Event()

    def fake_bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        if method == "GET" and "/api/v1/relay/capabilities" in url:
            return _fake_capabilities_response()
        if method == "POST" and url.endswith("/api/v1/relay/alerts"):
            posts.append((url, data))
            delivered.set()
            return _fake_event_accepted_response()
        raise AssertionError(f"unexpected relay request: {method} {url}")

    monkeypatch.setattr(evidence_export_client_module, "bounded_request", fake_bounded_request)

    serving = _FakeServingClient()
    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    runtime.run()

    assert [url for url, _data in posts] == ["http://relay.test/api/v1/relay/alerts"]
    posted = json.loads(posts[0][1] or b"{}")
    assert posted["edge_event_id"] == _EDGE_EVENT_ID
    assert posted["camera_id"] == "camera-a"
    assert posted["facility_id"] == "facility-a"


def test_worker_stop_joins_the_evidence_export_sender_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    outbox_path = _enable_export(monkeypatch, tmp_path)
    _stage_one_admitted_event(outbox_path)

    delivered = threading.Event()
    sender_threads: list[threading.Thread] = []

    def fake_bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        if method == "GET" and "/api/v1/relay/capabilities" in url:
            return _fake_capabilities_response()
        if method == "POST" and url.endswith("/api/v1/relay/alerts"):
            sender_threads.append(threading.current_thread())
            delivered.set()
            return _fake_event_accepted_response()
        raise AssertionError(f"unexpected relay request: {method} {url}")

    monkeypatch.setattr(evidence_export_client_module, "bounded_request", fake_bounded_request)

    serving = _FakeServingClient()
    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    runtime.run()

    assert len(sender_threads) == 1
    sender_thread = sender_threads[0]
    assert sender_thread.name == "evidence-sender"
    # run()'s finally block calls stop(), which calls stop_sender(): by the
    # time run() returns, the sender thread must already be joined and dead.
    assert not sender_thread.is_alive()


def test_per_camera_clip_recorder_views_are_distinct_objects_over_one_shared_recorder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every profile in PROFILE_REGISTRY resolves a real encoder (h264_nvenc or
    libx264); there is no "no encode support" case in this codebase, so this
    exercises the always-real-recorder composition path. ClipRecorder itself
    is one shared actor/encoder (the existing design), but composition must
    still hand each camera a distinct EventClipRecorder object."""
    _stub_heartbeat_transport(monkeypatch)
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))

    recorders: dict[str, object] = {}
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", "camera-b"), serving, loops, tmp_path)

    def capturing_factory(camera: CameraRuntimeConfig) -> object:
        recorder = runtime._default_clip_recorder(camera)  # noqa: SLF001
        recorders[camera.camera_id] = recorder
        return recorder

    runtime._clip_recorder_factory = capturing_factory  # noqa: SLF001

    runtime.run()

    assert runtime._clip_recorder is not None, "real ClipRecorder must have started"  # noqa: SLF001
    assert set(recorders) == {"camera-a", "camera-b"}
    first, second = recorders["camera-a"], recorders["camera-b"]
    assert first is not second
    assert isinstance(first, worker_module._CameraClipRecorderView)  # noqa: SLF001
    assert isinstance(second, worker_module._CameraClipRecorderView)  # noqa: SLF001
    # Distinct per-camera objects, same shared encoder/actor underneath --
    # ClipRecorder already keys mutable per-clip state by camera_id, so this
    # sharing never cross-contaminates camera state.
    assert first.recorder is second.recorder is runtime._clip_recorder  # noqa: SLF001
    assert first.camera_id == "camera-a"
    assert second.camera_id == "camera-b"


def test_build_clip_frame_feeder_binds_the_evidence_tap_and_the_shared_clip_recorder(
    tmp_path: Path,
) -> None:
    """Production gap A: ``_build_clip_frame_feeder`` must drain
    ``bus.evidence`` -- the subscription that exists for exactly one purpose,
    feeding clip recording, and that nothing else reads -- not
    ``bus.inference``, and must share the one process-wide ``ClipRecorder``
    ``_compose_evidence_export`` resolves once, not build a fresh recorder
    per camera."""
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)
    recorder = _FakeClipRecorder()
    runtime._clip_recorder = recorder  # type: ignore[assignment]  # noqa: SLF001
    bus = BoundedFrameBus()

    feeder = runtime._build_clip_frame_feeder("camera-a", bus)  # noqa: SLF001

    assert feeder is not None
    assert feeder.camera_id == "camera-a"
    assert feeder._subscription is bus.evidence  # noqa: SLF001
    assert feeder._subscription is not bus.inference  # noqa: SLF001
    assert feeder._recorder is recorder  # noqa: SLF001


def test_build_clip_frame_feeder_returns_none_when_clip_recording_never_started(
    tmp_path: Path,
) -> None:
    """Mirrors ``_compose_evidence_export``'s non-fatal degrade path: when the
    real ``ClipRecorder`` never started (``self._clip_recorder`` stays
    ``None``), there is no point draining a subscription nobody reads."""
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)
    assert runtime._clip_recorder is None  # noqa: SLF001  # sanity: unset before composition runs
    bus = BoundedFrameBus()

    feeder = runtime._build_clip_frame_feeder("camera-a", bus)  # noqa: SLF001

    assert feeder is None
