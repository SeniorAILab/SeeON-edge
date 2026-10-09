from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI

from backend.app.features.detection_settings.router import (
    DetectionSettingsPorts,
    detection_settings_ports,
)
from backend.app.lifespan import install_feature_ports


def test_getter_refuses_missing_ports_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="detection settings ports are not injected"):
        detection_settings_ports(app)
    assert not hasattr(app.state, "detection_settings_ports")


@pytest.mark.parametrize(
    ("wrong_ports", "error", "message"),
    [
        (None, RuntimeError, "detection settings ports are not injected"),
        (object(), TypeError, "detection settings ports have invalid type"),
        ({"camera_records": list}, TypeError, "detection settings ports have invalid type"),
    ],
)
def test_getter_refuses_wrong_ports_without_replacing_them(
    wrong_ports: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.detection_settings_ports = wrong_ports
    with pytest.raises(error, match=message):
        detection_settings_ports(app)
    assert app.state.detection_settings_ports is wrong_ports


def test_installed_ports_are_frozen_and_read_their_owners_on_each_call() -> None:
    app = FastAPI()
    install_feature_ports(app)
    ports = detection_settings_ports(app)
    assert isinstance(ports, DetectionSettingsPorts)
    with pytest.raises(FrozenInstanceError):
        ports.__setattr__("camera_records", list)
    with pytest.raises(RuntimeError, match="connection settings store is not injected"):
        ports.enrolled_facility_id()
    with pytest.raises(RuntimeError, match="camera registry is not injected"):
        ports.camera_records()
    app.state.camera_registry = object()
    with pytest.raises(TypeError, match="camera registry has invalid type"):
        ports.camera_records()
    assert detection_settings_ports(app) is ports
