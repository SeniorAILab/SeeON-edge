from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI

from backend.app.features.evidence.router import EvidencePorts, evidence_ports
from backend.app.lifespan import install_feature_ports


def test_getter_refuses_missing_ports_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="evidence ports are not injected"):
        evidence_ports(app)
    assert not hasattr(app.state, "evidence_ports")


@pytest.mark.parametrize(
    ("wrong_ports", "error", "message"),
    [
        (None, RuntimeError, "evidence ports are not injected"),
        (object(), TypeError, "evidence ports have invalid type"),
        ({"clip_export_enabled": dict}, TypeError, "evidence ports have invalid type"),
    ],
)
def test_getter_refuses_wrong_ports_without_replacing_them(
    wrong_ports: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.evidence_ports = wrong_ports
    with pytest.raises(error, match=message):
        evidence_ports(app)
    assert app.state.evidence_ports is wrong_ports


def test_installed_ports_are_frozen_and_read_the_runtime_settings_owner_on_each_call() -> None:
    app = FastAPI()
    install_feature_ports(app)
    ports = evidence_ports(app)
    assert isinstance(ports, EvidencePorts)
    with pytest.raises(FrozenInstanceError):
        ports.__setattr__("clip_export_enabled", dict)
    with pytest.raises(RuntimeError, match="runtime settings store is not injected"):
        ports.clip_export_enabled()
    app.state.runtime_settings_store = object()
    with pytest.raises(TypeError, match="runtime settings store has invalid type"):
        ports.clip_export_enabled()
    assert evidence_ports(app) is ports
