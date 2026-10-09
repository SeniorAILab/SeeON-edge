from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI

from backend.app.features.status.router import StatusPorts, status_ports
from backend.app.lifespan import install_feature_ports


def test_getter_refuses_missing_ports_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="status ports are not injected"):
        status_ports(app)
    assert not hasattr(app.state, "status_ports")


@pytest.mark.parametrize(
    ("wrong_ports", "error", "message"),
    [
        (None, RuntimeError, "status ports are not injected"),
        (object(), TypeError, "status ports have invalid type"),
        ({"runtime_settings": dict}, TypeError, "status ports have invalid type"),
    ],
)
def test_getter_refuses_wrong_ports_without_replacing_them(
    wrong_ports: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.status_ports = wrong_ports
    with pytest.raises(error, match=message):
        status_ports(app)
    assert app.state.status_ports is wrong_ports


def test_installed_ports_are_frozen_and_read_the_runtime_settings_owner_on_each_call() -> None:
    app = FastAPI()
    install_feature_ports(app)
    ports = status_ports(app)
    assert isinstance(ports, StatusPorts)
    with pytest.raises(FrozenInstanceError):
        ports.__setattr__("runtime_settings", dict)
    with pytest.raises(RuntimeError, match="runtime settings store is not injected"):
        ports.runtime_settings()
    app.state.runtime_settings_store = object()
    with pytest.raises(TypeError, match="runtime settings store has invalid type"):
        ports.runtime_settings()
    assert status_ports(app) is ports
