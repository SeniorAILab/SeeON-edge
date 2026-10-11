from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI

from backend.app.bootstrap.lifespan import install_feature_ports
from backend.app.features.connection.router import (
    ConnectionPorts,
    _kick_backend_config_refresh,
    connection_ports,
)


def test_getter_refuses_missing_ports_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="connection ports are not injected"):
        connection_ports(app)
    assert not hasattr(app.state, "connection_ports")


@pytest.mark.parametrize(
    ("wrong_ports", "error", "message"),
    [
        (None, RuntimeError, "connection ports are not injected"),
        (object(), TypeError, "connection ports have invalid type"),
        ({"apply_settings": lambda: None}, TypeError, "connection ports have invalid type"),
    ],
)
def test_getter_refuses_wrong_ports_without_replacing_them(
    wrong_ports: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.connection_ports = wrong_ports
    with pytest.raises(error, match=message):
        connection_ports(app)
    assert app.state.connection_ports is wrong_ports


def test_installed_ports_are_frozen_and_stable() -> None:
    app = FastAPI()
    install_feature_ports(app)
    ports = connection_ports(app)
    assert isinstance(ports, ConnectionPorts)
    with pytest.raises(FrozenInstanceError):
        ports.__setattr__("apply_settings", lambda: None)
    assert connection_ports(app) is ports


def test_kick_submits_the_injected_refresh_to_the_refresh_executor() -> None:
    app = FastAPI()
    calls: list[str] = []

    def refresh() -> bool:
        calls.append("refresh")
        return True

    app.state.connection_ports = ConnectionPorts(
        apply_settings=lambda: None, refresh_backend_config=refresh
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.backend_config_refresh_executor = executor
        _kick_backend_config_refresh(app)
    assert calls == ["refresh"]


def test_kick_is_a_noop_without_a_refresh_executor() -> None:
    app = FastAPI()
    _kick_backend_config_refresh(app)
    app.state.backend_config_refresh_executor = None
    _kick_backend_config_refresh(app)
    assert not hasattr(app.state, "connection_ports")


def test_kick_tolerates_a_shut_down_executor() -> None:
    app = FastAPI()
    calls: list[str] = []

    def refresh() -> bool:
        calls.append("refresh")
        return True

    app.state.connection_ports = ConnectionPorts(
        apply_settings=lambda: None, refresh_backend_config=refresh
    )
    executor = ThreadPoolExecutor(max_workers=1)
    executor.shutdown()
    app.state.backend_config_refresh_executor = executor
    _kick_backend_config_refresh(app)
    assert calls == []


def test_kick_refuses_missing_ports_when_an_executor_is_present() -> None:
    app = FastAPI()
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.backend_config_refresh_executor = executor
        with pytest.raises(RuntimeError, match="connection ports are not injected"):
            _kick_backend_config_refresh(app)
