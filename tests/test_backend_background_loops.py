from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import lifespan as lifespan_module
from backend.app.edge_db.postgres import PostgresError
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.status.heartbeat_store import HeartbeatStore
from backend.app.main import create_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox", "tests_support.postgres_app_env")


class FlakyDatabase:
    def __init__(self, database: object, failures: int) -> None:
        self.database = database
        self.failures = failures

    def read(self, operation: Callable[..., object]) -> object:
        if self.failures > 0:
            self.failures -= 1
            raise PostgresError("connection lost during registry read")
        return self.database.read(operation)


class RecordingIngestClient:
    def __init__(self, calls: list[str] | None = None, camera_id: str | None = None) -> None:
        self.calls = [] if calls is None else calls
        self.camera_id = camera_id

    def for_camera(self, camera_id: str) -> RecordingIngestClient:
        return RecordingIngestClient(self.calls, camera_id)

    def send_heartbeat(self) -> bool:
        assert self.camera_id is not None
        self.calls.append(self.camera_id)
        return True


async def _run_until(
    loop_factory: Callable[[asyncio.Event, ThreadPoolExecutor], Coroutine[object, object, None]],
    done: Callable[[], bool],
) -> asyncio.Task[None]:
    stop = asyncio.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(loop_factory(stop, executor))
    try:
        for _ in range(200):
            if done() or task.done():
                break
            await asyncio.sleep(0.01)
    finally:
        stop.set()
        await asyncio.wait({task}, timeout=1.0)
        executor.shutdown(wait=True)
    return task


def test_heartbeat_relay_loop_relays_on_the_next_tick_after_a_failed_registry_read(
    postgres_product_sandbox: ProductSandbox,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = CameraRegistryStore(
        postgres_product_sandbox.database, postgres_product_sandbox.authority
    )
    registry.create(
        camera_id="cam-online",
        label="cam-online",
        rtsp_url="rtsp://cam-online",
        space_id=None,
        status="offline",
        backend_camera_id="cam-online",
    )
    registry.database = FlakyDatabase(registry.database, failures=1)
    heartbeat_store = HeartbeatStore(stale_after_sec=90.0)
    heartbeat_store.record("cam-online", "fac-1")
    client = RecordingIngestClient()
    app = SimpleNamespace(
        state=SimpleNamespace(
            heartbeat_store=heartbeat_store,
            camera_registry=registry,
            backend_ingest_client=client,
        )
    )

    with caplog.at_level(logging.ERROR, logger="backend.app.lifespan"):
        task = asyncio.run(
            _run_until(
                lambda stop, executor: lifespan_module._backend_heartbeat_relay_loop(
                    app,
                    stop,
                    executor,
                    0.01,
                ),
                lambda: bool(client.calls),
            )
        )

    assert client.calls[:1] == ["cam-online"]
    assert not task.cancelled() and task.exception() is None
    [record] = [r for r in caplog.records if r.name == "backend.app.lifespan"]
    assert record.getMessage() == "backend heartbeat relay tick failed"
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "connection lost during registry read"


def test_config_refresh_loop_refreshes_on_the_next_tick_after_a_failed_refresh(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    def refresh(app: object, stop_token: asyncio.Event | None = None) -> bool:
        calls.append("refresh")
        if len(calls) == 1:
            raise PostgresError("connection lost during roster resume")
        return True

    monkeypatch.setattr(lifespan_module, "refresh_backend_config", refresh)
    monkeypatch.setattr(lifespan_module, "_backend_config_refresh_sec", lambda: 0.01)
    app = SimpleNamespace(state=SimpleNamespace())

    with caplog.at_level(logging.ERROR, logger="backend.app.lifespan"):
        task = asyncio.run(
            _run_until(
                lambda stop, executor: lifespan_module._backend_config_refresh_loop(
                    app,
                    stop,
                    executor,
                ),
                lambda: len(calls) >= 2,
            )
        )

    assert calls[:2] == ["refresh", "refresh"]
    assert not task.cancelled() and task.exception() is None
    [record] = [r for r in caplog.records if r.name == "backend.app.lifespan"]
    assert record.getMessage() == "backend config refresh tick failed"
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "connection lost during roster resume"


@pytest.mark.usefixtures("postgres_app_env")
def test_shutdown_finishes_cleanup_when_a_background_task_already_died(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def dead_refresh_loop(
        app: object, stop_event: asyncio.Event, executor: ThreadPoolExecutor
    ) -> None:
        raise RuntimeError("config refresh loop died")

    monkeypatch.setattr(lifespan_module, "_backend_config_refresh_loop", dead_refresh_loop)
    app = create_app()

    with caplog.at_level(logging.ERROR, logger="backend.app.lifespan"):
        with TestClient(app):
            assert app.state.backend_heartbeat_relay_task is not None

    assert app.state.backend_config_refresh_task is None
    assert app.state.backend_config_refresh_executor is None
    assert app.state.backend_heartbeat_relay_task is None
    assert app.state.backend_heartbeat_relay_executor is None
    [record] = [r for r in caplog.records if r.name == "backend.app.lifespan"]
    assert record.getMessage() == "background task backend-config-refresh ended with an error"
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "config refresh loop died"
