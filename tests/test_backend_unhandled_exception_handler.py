import logging

import psycopg
import pytest
from fastapi import APIRouter
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
    records = [r for r in caplog.records if r.name == "backend.app.main"]
    [record] = records
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
