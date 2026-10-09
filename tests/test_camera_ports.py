from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI

from backend.app.features.cameras.dependencies import CameraPorts, camera_ports
from backend.app.features.status.heartbeat_store import DEFAULT_STALE_AFTER_SEC, HeartbeatStore
from backend.app.lifespan import install_feature_ports


def test_getter_refuses_missing_ports_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="camera ports are not injected"):
        camera_ports(app)
    assert not hasattr(app.state, "camera_ports")


@pytest.mark.parametrize(
    ("wrong_ports", "error", "message"),
    [
        (None, RuntimeError, "camera ports are not injected"),
        (object(), TypeError, "camera ports have invalid type"),
        ({"enrolled_facility_id": str}, TypeError, "camera ports have invalid type"),
    ],
)
def test_getter_refuses_wrong_ports_without_replacing_them(
    wrong_ports: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.camera_ports = wrong_ports
    with pytest.raises(error, match=message):
        camera_ports(app)
    assert app.state.camera_ports is wrong_ports


def test_installed_ports_are_frozen_and_read_the_connection_owner_on_each_call() -> None:
    app = FastAPI()
    install_feature_ports(app)
    ports = camera_ports(app)
    assert isinstance(ports, CameraPorts)
    with pytest.raises(FrozenInstanceError):
        ports.__setattr__("enrolled_facility_id", str)
    with pytest.raises(RuntimeError, match="connection settings store is not injected"):
        ports.enrolled_facility_id()
    assert not hasattr(app.state, "heartbeat_store")
    assert ports.heartbeats() == {"cameras": {}, "stale_after_sec": DEFAULT_STALE_AFTER_SEC}
    assert isinstance(app.state.heartbeat_store, HeartbeatStore)
    with pytest.raises(RuntimeError, match="runtime settings store is not injected"):
        ports.clip_export_setting()
    assert camera_ports(app) is ports
