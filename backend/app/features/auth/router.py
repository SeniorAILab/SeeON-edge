"""Server-side dashboard login and logout routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.shared.dashboard_auth import (
    DASHBOARD_SESSION_COOKIE,
    authorize_dashboard,
    dashboard_sessions,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class DashboardLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


@router.post("/session", status_code=status.HTTP_204_NO_CONTENT)
def login(payload: DashboardLoginRequest, request: Request, response: Response) -> None:
    sessions = dashboard_sessions(request)
    if sessions is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard credentials are not configured",
        )
    token = sessions.authenticate(payload.username, payload.password)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid dashboard credentials",
        )
    response.set_cookie(
        DASHBOARD_SESSION_COOKIE,
        token,
        max_age=sessions.ttl_seconds,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


@router.get("/session", status_code=status.HTTP_204_NO_CONTENT)
def session(request: Request) -> None:
    authorize_dashboard(request)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response) -> None:
    sessions = dashboard_sessions(request)
    if sessions is not None:
        sessions.revoke(request.cookies.get(DASHBOARD_SESSION_COOKIE))
    response.delete_cookie(DASHBOARD_SESSION_COOKIE, path="/", samesite="strict")


__all__ = ["router"]
