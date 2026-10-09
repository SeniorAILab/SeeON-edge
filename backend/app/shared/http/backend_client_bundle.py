from __future__ import annotations

from fastapi import FastAPI

from backend.app.shared.backend_client_bundle import BackendClientBundle


def backend_client_bundle(app: FastAPI) -> BackendClientBundle | None:
    state = getattr(app, "state", None)
    candidate = getattr(state, "backend_client_bundle", None)
    return candidate if isinstance(candidate, BackendClientBundle) else None


__all__ = ["backend_client_bundle"]
