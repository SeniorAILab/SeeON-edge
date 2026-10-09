import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app import lifespan
from backend.app.features.clips import catalog_indexer
from worker.adapters.model.errors import FatalAcceleratorError
from worker.pipeline.decision import incident_manager
from worker.pipeline.decision.incident_manager import IncidentManager
from worker.runtime import worker as worker_runtime
from worker.runtime.clip_analysis_lifecycle import retry_teardown
from worker.runtime.worker import NativeHeartbeatLoop, WorkerRuntime
from worker.types import BusinessEvent

SECRET = "token=hunter2"


def visible(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.INFO]


def assert_contained_without_text(
    caplog: pytest.LogCaptureFixture, stage: str, *, level: int = logging.WARNING
) -> list[logging.LogRecord]:
    records = visible(caplog)
    assert not [r for r in records if r.exc_info]
    assert SECRET not in "".join(r.getMessage() for r in records)
    contained = [r for r in records if f"stage={stage} " in f"{r.getMessage()} "]
    assert contained
    assert {r.levelno for r in contained if "recovered" not in r.getMessage()} == {level}
    return contained


class FailingThenOk:
    def __init__(self, failures: int, on_last: object = None) -> None:
        self.failures = failures
        self.calls = 0
        self.on_last = on_last

    def __call__(self, *args: object) -> object:
        self.calls += 1
        if self.calls > self.failures:
            if callable(self.on_last):
                self.on_last()
            return SimpleNamespace(remaining=0, retry_after_seconds=None)
        raise RuntimeError(SECRET)


def test_clip_catalog_reconcile_failures_are_throttled_and_recovery_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> FailingThenOk:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        reconcile = FailingThenOk(10, lambda: loop.call_soon_threadsafe(stop.set))
        indexer = SimpleNamespace(reconcile=reconcile)
        with ThreadPoolExecutor(1) as executor:
            await catalog_indexer.run_clip_catalog_indexer(
                indexer, object(), stop, executor, 0.0, 1
            )
        return reconcile

    with caplog.at_level(logging.INFO):
        reconcile = asyncio.run(scenario())
    assert reconcile.calls == 11
    contained = assert_contained_without_text(
        caplog, "clip_catalog_reconcile", level=logging.ERROR
    )
    assert [r.levelno for r in contained] == [logging.ERROR, logging.INFO]
    assert "failures=10" in contained[-1].getMessage()


def test_backend_outbox_sender_tick_failures_are_throttled_and_keep_looping(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lifespan, "sender_delay", lambda *args: 0.0)

    async def scenario() -> FailingThenOk:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        send = FailingThenOk(10, lambda: loop.call_soon_threadsafe(stop.set))
        monkeypatch.setattr(lifespan, "send_outbox_once", send)
        app = SimpleNamespace(
            state=SimpleNamespace(
                backend_outbox_sender_status=SimpleNamespace(consecutive_failures=0)
            )
        )
        with ThreadPoolExecutor(1) as executor:
            await lifespan._backend_outbox_sender_loop(app, stop, executor)
        return send

    with caplog.at_level(logging.INFO):
        send = asyncio.run(scenario())
    assert send.calls == 11
    contained = assert_contained_without_text(
        caplog, "backend_outbox_sender_tick", level=logging.ERROR
    )
    assert [r.levelno for r in contained] == [logging.ERROR, logging.INFO]


def event(identity: str = "onset") -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall",
        identity=identity,
        camera_id="room-camera",
        facility_id="facility-1",
        time_sec=100.0,
        probability=0.91,
    )


def test_an_unusable_identity_journal_falls_back_and_logs_an_error_without_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = incident_manager.EventIdentityStore

    def store(path: Path | None) -> object:
        if path is not None:
            raise OSError(SECRET)
        return real(None)

    monkeypatch.setattr(incident_manager, "EventIdentityStore", store)
    with caplog.at_level(logging.INFO):
        manager = IncidentManager(cooldown_sec=0.0, identity_path=tmp_path / "id.jsonl")
    assert manager.identity_journal_failures == 1
    assert manager.admit(event(), now_sec=100.0) is not None
    assert_contained_without_text(caplog, "event_identity_journal_open", level=logging.ERROR)


def test_identity_resolve_failures_admit_fresh_ids_and_are_throttled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manager = IncidentManager(cooldown_sec=0.0, identity_path=tmp_path / "id.jsonl")

    def explode(_key: str) -> str:
        raise OSError(SECRET)

    manager._identities.resolve = explode
    with caplog.at_level(logging.INFO):
        admitted = [manager.admit(event(f"onset-{n}"), now_sec=100.0 + n) for n in range(10)]
    assert all(a is not None and str(a.identity) for a in admitted)
    assert len({str(a.identity) for a in admitted if a is not None}) == 10
    assert manager.identity_journal_failures == 10
    contained = assert_contained_without_text(
        caplog, "event_identity_journal_resolve", level=logging.ERROR
    )
    assert len(contained) == 1
    assert "camera_id=room-camera" in contained[0].getMessage()


class FlakyTerminate:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, process: object) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(SECRET)


def test_clip_analysis_teardown_retry_logs_each_attempt_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    terminate = FlakyTerminate(2)
    with caplog.at_level(logging.INFO):
        assert retry_teardown(object(), terminate, attempts=3) is True
    contained = assert_contained_without_text(caplog, "clip_analysis_teardown", level=logging.ERROR)
    assert len(contained) == 2


def test_clip_ready_notify_failure_is_contained_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def notify(*args: object, **kwargs: object) -> None:
        raise RuntimeError(SECRET)

    runtime = object.__new__(WorkerRuntime)
    runtime._clip_analysis_supervisor = SimpleNamespace(notify=notify)
    publication = SimpleNamespace(
        clip_id="clip-9", video_path=Path("v.mp4"), sha256="a" * 64, size_bytes=1, duration_ms=1
    )
    with caplog.at_level(logging.INFO):
        assert runtime._on_clip_ready(publication) is None
    [record] = assert_contained_without_text(caplog, "clip_analysis_notify")
    assert "clip_id=clip-9" in record.getMessage()


def test_replay_sealed_clips_counts_failures_and_logs_error_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing() -> None:
        raise RuntimeError(SECRET)

    bindings = [
        SimpleNamespace(camera_id="camera-a", replay_sealed=failing),
        SimpleNamespace(camera_id="camera-b", replay_sealed=lambda: None),
    ]
    with caplog.at_level(logging.INFO):
        assert WorkerRuntime._replay_sealed_clips(bindings) == 1
    [record] = assert_contained_without_text(
        caplog, "sealed_clip_replay_binding", level=logging.ERROR
    )
    assert "camera_id=camera-a" in record.getMessage()


class CountingPump:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.reads = 0

    @property
    def processed_count(self) -> int:
        self.reads += 1
        return self.reads


class Reporter:
    def __init__(self, error: BaseException | None, stop_after: int, loop_ref: list) -> None:
        self.error = error
        self.calls = 0
        self.stop_after = stop_after
        self.loop_ref = loop_ref

    def mark_ready(self, camera_id: str) -> None:
        self.calls += 1
        if self.calls >= self.stop_after:
            self.loop_ref[0].stop()
        if self.error is not None:
            raise self.error


def heartbeat_loop(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, stop_after: int
) -> tuple[NativeHeartbeatLoop, Reporter]:
    loop_ref: list = []
    reporter = Reporter(error, stop_after, loop_ref)
    monkeypatch.setattr(worker_runtime, "HeartbeatReporter", lambda worker, camera: reporter)
    loop = NativeHeartbeatLoop(
        object(), [SimpleNamespace(camera_id="camera-a")], [CountingPump("camera-a")], tick_sec=0.0
    )
    loop_ref.append(loop)
    return loop, reporter


def test_native_heartbeat_failures_are_throttled_per_camera(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    loop, reporter = heartbeat_loop(monkeypatch, ConnectionError(SECRET), stop_after=10)
    with caplog.at_level(logging.INFO):
        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert reporter.calls == 10
    contained = assert_contained_without_text(caplog, "native_heartbeat")
    assert len(contained) == 1
    assert "camera_id=camera-a" in contained[0].getMessage()


def test_native_heartbeat_still_propagates_a_fatal_accelerator_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _ = heartbeat_loop(monkeypatch, FatalAcceleratorError("gpu lost"), stop_after=99)
    with pytest.raises(FatalAcceleratorError):
        loop.run()
