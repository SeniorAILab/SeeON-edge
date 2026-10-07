import logging
import socket
import threading
import time
import urllib.error
import urllib.request

import psycopg
import pytest
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import PostgresError
from backend.app.main import create_app

SECRET = "postgresql://admin:hunter2@db/edge"


def client_raising(error: Exception) -> TestClient:
    app = create_app(lifespan=None)
    router = APIRouter()

    @router.get("/__boom")
    def boom() -> None:
        raise error

    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def test_an_unhandled_exception_returns_a_bare_500_without_internals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = client_raising(RuntimeError(SECRET))
    with caplog.at_level(logging.ERROR, logger="backend.app.main"):
        response = client.get("/__boom")
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert SECRET not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text
    [record] = [r for r in caplog.records if r.name == "backend.app.main"]
    assert record.getMessage() == (
        "unhandled request failure method=GET path=/__boom exception_class=RuntimeError"
    )
    assert record.exc_info is not None
    assert record.exc_info[1].args == (SECRET,)


def test_every_unhandled_exception_gets_the_same_body() -> None:
    first = client_raising(ValueError("a")).get("/__boom")
    second = client_raising(KeyError("b")).get("/__boom")
    assert first.status_code == second.status_code == 500
    assert first.json() == second.json() == {"detail": "internal server error"}


@pytest.mark.parametrize(
    "error",
    [PostgresError("db down"), psycopg.OperationalError("db down")],
    ids=["PostgresError", "psycopg.Error"],
)
def test_the_existing_503_mapping_still_wins(error: Exception) -> None:
    response = client_raising(error).get("/__boom")
    assert response.status_code == 503


def serve_and_get(app: FastAPI, path: str) -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                return int(response.status)
        except urllib.error.HTTPError as error:
            return int(error.code)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_under_uvicorn_the_traceback_is_logged_by_the_handler_and_again_by_uvicorn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = client_raising(RuntimeError(SECRET))
    with caplog.at_level(logging.INFO):
        status_code = serve_and_get(client.app, "/__boom")
    assert status_code == 500
    tracebacks = [record for record in caplog.records if record.exc_info]
    assert [record.name for record in tracebacks] == ["backend.app.main", "uvicorn.error"]
    assert tracebacks[0].getMessage() == (
        "unhandled request failure method=GET path=/__boom exception_class=RuntimeError"
    )
    assert tracebacks[1].getMessage() == "Exception in ASGI application\n"
